"""Utility class for final output summarizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from portia.introspection_agents.introspection_agent import (
    COMPLETED_OUTPUT,
    SKIPPED_OUTPUT,
)
from portia.model import Message

if TYPE_CHECKING:
    from portia.config import Config
    from portia.plan import Plan
    from portia.plan_run import PlanRun


class FinalOutputSummarizer:
    """Utility class responsible for summarizing the run outputs for final output's summary.

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
            config (Config): The configuration for the llm.

        """
        self.config = config

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
                output_value = (
                    outputs[step.output].get_summary()
                    if outputs[step.output].get_value()
                    in (
                        SKIPPED_OUTPUT,
                        COMPLETED_OUTPUT,
                    )
                    else outputs[step.output].get_value()
                )
                context.append(f"Task: {step.task}")
                context.append(f"Output: {output_value}")
                context.append("----------")
        return "\n".join(context)

    def create_summary(self, plan: Plan, plan_run: PlanRun) -> str | None:
        """Execute the summarizer llm and return the summary as a string.

        Args:
            plan (Plan): The plan containing the steps.
            plan_run (PlanRun): The run to summarize.

        Returns:
            str | None: The generated summary or None if generation fails.

        """
        model = self.config.get_summarizer_model()
        context = self._build_tasks_and_outputs_context(plan, plan_run)
        response = model.get_response(
            [Message(content=self.SUMMARIZE_TASK + context, role="user")],
        )
        return str(response.content) if response.content else None
