"""Summarizer agent implementation."""

from portia.agents.base_agent import Output
from portia.agents.context import build_tasks_and_outputs_context
from portia.config import Config
from portia.llm_wrapper import LLMWrapper
from portia.plan import Plan
from portia.workflow import Workflow


class SummarizerAgent:
    """Agent responsible for generating summaries of workflow outputs."""

    SUMMARIZE_TASK = (
        "Summarize all tasks and outputs. Make sure the "
        "summary is including all the previous tasks and outputs and biased towards "
        "the last step output of the plan. Your summary "
        "should be concise and to the point with maximum 500 characters. Do not "
        "include 'Summary:' in the beginning of the summary.\n"
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

    def execute_sync(self) -> Output:
        """Execute the summarization.

        Returns:
            Output: The generated summary

        """
        llm = LLMWrapper(self.config).to_langchain()
        context = build_tasks_and_outputs_context(self.plan, self.workflow)
        response = llm.invoke(
            self.SUMMARIZE_TASK + context,
        )
        summary_content = str(response.content) if response.content is not None else None
        return Output(
            value=summary_content,
            summary=summary_content,
        )

