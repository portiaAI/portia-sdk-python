"""StepSummarizer implementation.

The StepSummarizer can be used by agents to summarize the output of a given tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jinja2 import Template
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain_core.messages import ToolMessage
from langgraph.graph import MessagesState  # noqa: TC002

from portia.execution_agents.output import Output
from portia.logger import logger
from portia.model import GenerativeModel, Message
from portia.planning_agents.context import get_tool_descriptions_for_tools

if TYPE_CHECKING:
    from portia.config import Config
    from portia.model import GenerativeModel
    from portia.plan import Step
    from portia.tool import Tool


class StepSummarizer:
    """Class to summarize the output of a tool using llm.

    This is used only on the tool output message.

    Attributes:
        summarizer_prompt (ChatPromptTemplate): The prompt template used to generate the summary.
        model (GenerativeModel): The language model used for summarization.
        summary_max_length (int): The maximum length of the summary.
        step (Step): The step that produced the output.

    """

    summarizer_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a highly skilled summarizer. Your task is to create a textual summary"
                    "of the provided tool output, make sure to follow the guidelines provided.\n"
                    "- Focus on the key information and maintain accuracy.\n"
                    "- Make sure to not exceed the max limit of {max_length} characters.\n"
                    "- Don't produce an overly long summary if it doesn't make sense.\n"
                ),
            ),
            HumanMessagePromptTemplate.from_template(
                "Here is original task:\n{task_description}\n"
                "Here is the description of the tool that produced "
                "the output:\n{tool_description}\n"
                "Please summarize the following output:\n{tool_output}\n",
            ),
        ],
    )

    def __init__(
        self,
        config: Config,
        model: GenerativeModel,
        tool: Tool,
        step: Step,
        summary_max_length: int = 500,
    ) -> None:
        """Initialize the model.

        Args:
            config (Config): The configuration for the run.
            model (GenerativeModel): The language model used for summarization.
            tool (Tool): The tool used for summarization.
            step (Step): The step that produced the output.
            summary_max_length (int): The maximum length of the summary. Default is 500 characters.

        """
        self.config = config
        self.model = model
        self.summary_max_length = summary_max_length
        self.tool = tool
        self.step = step

    def invoke(self, state: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state.

        This method processes the last message in the state, checks if it's a tool message with an
        output, and if so, generates a summary of the tool's output. The summary is then added to
        the artifact of the last message.

        Args:
            state (MessagesState): The current state of the messages, which includes the output.

        Returns:
            dict[str, Any]: A dict containing the updated message state, including the summary.

        Raises:
            Exception: If an error occurs during the invocation of the summarizer model.

        """
        messages = state["messages"]
        last_message = messages[-1] if len(messages) > 0 else None
        if not isinstance(last_message, ToolMessage) or not isinstance(
            last_message.artifact,
            Output,
        ):
            return {"messages": [last_message]}

        logger().debug(f"Invoke SummarizerModel on the tool output of {last_message.name}.")
        tool_output = last_message.content
        if self.config.exceeds_output_threshold(tool_output):
            tool_output = (
                f"This is a large value (full length: {len(str(tool_output))} characters) - it is "
                "too long to provide the full value, but it starts with:"
                f"{self._truncate(tool_output, self.config.large_output_threshold_tokens)}"
            )

        try:
            response: Message = self.model.get_response(
                messages=[
                    Message.from_langchain(m)
                    for m in self.summarizer_prompt.format_messages(
                        tool_output=tool_output,
                        max_length=self.summary_max_length,
                        tool_description=get_tool_descriptions_for_tools([self.tool]),
                        task_description=self.step.task,
                    )
                ],
            )
            summary = response.content
            last_message.artifact.summary = summary  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001 - we want to catch all exceptions
            logger().error("Error in SummarizerModel invoke (Skipping summaries): " + str(e))

        return {"messages": [last_message]}

    def _truncate(self, content: str | dict | list[str | dict], max_len_chars: int) -> str:
        """Truncate a value so it is no longer than max_len_chars."""
        content_str = Template("{{ content }}").render(content=str(content))
        return content_str[:max_len_chars]
