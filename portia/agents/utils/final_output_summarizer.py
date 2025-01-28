"""Summarizer agent implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from portia.llm_wrapper import LLMWrapper

if TYPE_CHECKING:
    from portia.config import Config
    from portia.plan import Plan
    from portia.workflow import Workflow


class FinalOutputSummarizer:
    """Agent responsible for generating summaries of workflow outputs."""

    SUMMARIZE_TASK = (
        "Summarize all tasks and outputs that answers the query given. Make sure the "
        "summary is including all the previous tasks and outputs and biased towards "
        "the last step output of the plan. Your summary "
        "should be concise and to the point with maximum 500 characters. Do not "
        "include 'Summary:' in the beginning of the summary. Do not make up information "
        "not used in the context.\n"
    )

    def __init__(self, workflow: Workflow, plan: Plan, config: Config) -> None:
        """Initialize the summarizer agent.

        Args:
            workflow (Workflow): The workflow to summarize
            plan (Plan): The plan containing the steps
            config: The configuration for the agent

        """
        self.plan = plan
        self.workflow = workflow
        self.config = config

    def build_tasks_and_outputs_context(self) -> str:
        """Get the tasks and outputs context."""
        context = []
        context.append(f"Query: {self.plan.plan_context.query}")
        context.append("----------")
        for step in self.plan.steps:
            outputs = self.workflow.outputs.step_outputs
            if step.output in outputs:
                context.append(f"Task: {step.task}")
                context.append(f"Output: {outputs[step.output].value}")
                context.append("----------")
        return "\n".join(context)

    def create_summary(self) -> str | None:
        """Execute the summarizer llm and return the summary as an output.

        Returns:
            Output: The generated summary.

        """
        llm = LLMWrapper(self.config).to_langchain()
        context = self.build_tasks_and_outputs_context()
        response = llm.invoke(
            self.SUMMARIZE_TASK + context,
        )
        return str(response.content) if response.content is not None else None
