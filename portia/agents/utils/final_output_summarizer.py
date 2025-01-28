"""Summarizer agent implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from portia.llm_wrapper import LLMWrapper

if TYPE_CHECKING:
    from portia.config import Config
    from portia.plan import Plan
    from portia.workflow import Workflow


class FinalOutputSummarizer:
    """Summarizer responsible for summarizing the workflow outputs for final output's summary.

    Attributes:
        config (Config): The configuration for the llm.

    """

    SUMMARIZE_TASK = (
        "Summarize all tasks and outputs that answers the query given. Make sure the "
        "summary is including all the previous tasks and outputs and biased towards "
        "the last step output of the plan. Your summary "
        "should be concise and to the point with maximum 500 characters. Do not "
        "include 'Summary:' in the beginning of the summary. Do not make up information "
        "not used in the context.\n"
    )

    def __init__(self, config: Config) -> None:
        """Initialize the summarizer agent.

        Args:
            config(Config): The configuration for the agent

        """
        self.config = config

    def _build_tasks_and_outputs_context(self, plan: Plan, workflow: Workflow) -> str:
        """Build the query, tasks and outputs context.

        Args:
            plan(Plan): The plan containing the steps
            workflow(Workflow): The workflow to get the outputs from.

        Returns:
            str: The formatted context string

        """
        context = []
        context.append(f"Query: {plan.plan_context.query}")
        context.append("----------")
        for step in plan.steps:
            outputs = workflow.outputs.step_outputs
            if step.output in outputs:
                context.append(f"Task: {step.task}")
                context.append(f"Output: {outputs[step.output].value}")
                context.append("----------")
        return "\n".join(context)

    def create_summary(self, plan: Plan, workflow: Workflow) -> str | None:
        """Execute the summarizer llm and return the summary as a string.

        Args:
            plan (Plan): The plan containing the steps
            workflow (Workflow): The workflow to summarize

        Returns:
            str | None: The generated summary or None if generation fails

        """
        llm = LLMWrapper(self.config).to_langchain()
        context = self._build_tasks_and_outputs_context(plan, workflow)
        response = llm.invoke(
            self.SUMMARIZE_TASK + context,
        )
        return str(response.content) if response.content else None
