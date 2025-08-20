"""Utility class for final output summarizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from portia.introspection_agents.introspection_agent import (
    COMPLETED_OUTPUT,
    SKIPPED_OUTPUT,
)
from portia.logger import logger
from portia.model import Message
from portia.token_check import exceeds_context_threshold

if TYPE_CHECKING:
    from portia.config import Config
    from portia.execution_agents.output import Output
    from portia.plan import Plan
    from portia.plan_run import PlanRun
    from portia.storage import AgentMemory


class FinalOutputSummarizer:
    """Utility class responsible for summarizing the run outputs for final output's summary.

    Attributes:
        config (Config): The configuration for the llm.
        agent_memory (AgentMemory): The agent memory to use for the summarizer.

    """

    summarizer_only_prompt = (
        "Summarize all tasks and outputs that answers the query given. Make sure the "
        "summary is including all the previous tasks and outputs and biased towards "
        "the last step output of the plan. Your summary "
        "should be concise and to the point with maximum 500 characters. Do not "
        "include 'Summary:' in the beginning of the summary. Do not make up information "
        "not used in the context.\n"
    )

    summarizer_and_structured_output_prompt = (
        summarizer_only_prompt
        + "The output should also include the structured output of the plan run as specified to "
        "the output schema.\n"
    )

    def __init__(self, config: Config, agent_memory: AgentMemory) -> None:
        """Initialize the summarizer agent.

        Args:
            config (Config): The configuration for the llm.
            agent_memory (AgentMemory): The agent memory to use for the summarizer.

        """
        self.config = config
        self.agent_memory = agent_memory

    def _build_tasks_and_outputs_context(self, plan: Plan, plan_run: PlanRun) -> str:
        """Build the query, tasks and outputs context.

        Args:
            plan(Plan): The plan containing the steps.
            plan_run(PlanRun): The run to get the outputs from.

        Returns:
            str: The formatted context string

        """
        context = []
        context.append(f"Query: {plan.plan_context.query}")
        context.append("----------")
        outputs = plan_run.outputs.step_outputs
        for step in plan.steps:
            if step.output in outputs:
                output_value = self.get_output_value(outputs[step.output])
                context.append(f"Task: {step.task}")
                context.append(f"Output: {output_value}")
                context.append("----------")
        return "\n".join(context)

    def get_output_value(self, output: Output) -> str | None:
        """Get the value to use for the specified output.

        This ensures that introspection outputs and outputs that are too large for the LLM context
        window are handled correctly.
        """
        value = output.full_value(self.agent_memory)
        if value in (SKIPPED_OUTPUT, COMPLETED_OUTPUT):
            return output.get_summary()
        if exceeds_context_threshold(value, self.config.get_summarizer_model(), 0.9):
            return output.get_summary()
        return value

    def create_summary(self, plan: Plan, plan_run: PlanRun) -> str | BaseModel | None:
        """Execute the summarizer llm and return the summary as a string.

        Args:
            plan (Plan): The plan containing the steps.
            plan_run (PlanRun): The run to summarize.

        Returns:
            str | BaseModel | None: The generated summary or None if generation fails.

        """
        model = self.config.get_summarizer_model()
        context = self._build_tasks_and_outputs_context(plan, plan_run)
        if plan_run.structured_output_schema:

            class SchemaWithSummary(plan_run.structured_output_schema):
                # fo_summary prepended here with fo (=final_output) so as not to conflict with any
                # existing fields in the structured output schema if the user was to want to supply
                # their own summary field with a description of what to summarize, e.g.
                # summary = Field(description="The summary of the weather in london") # noqa: ERA001
                fo_summary: str = Field(description="The summary of the plan output")

            msg = self.summarizer_and_structured_output_prompt + context
            logger().trace("LLM call: summarization (final output)")
            return model.get_structured_response(
                [Message(content=msg, role="user")],
                SchemaWithSummary,
            )
        msg = self.summarizer_only_prompt + context
        logger().trace("LLM call: summarization (final output)")
        response = model.get_response([Message(content=msg, role="user")])
        return str(response.content) if response.content else None
