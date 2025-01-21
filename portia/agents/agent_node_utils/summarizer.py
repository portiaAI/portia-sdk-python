"""Summarizer model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain_core.messages import BaseMessage, ToolMessage

from portia.agents.base_agent import Output
from portia.logger import logger

if TYPE_CHECKING:
    from langchain.chat_models.base import BaseChatModel
    from langgraph.graph import MessagesState

class LLMSummarizer:
    """Model to generate a summary for the textual output of a tool.

    This model is used only on the tool output message.
    """

    summarizer_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=(
                "You are a highly skilled summarizer. Your task is to create a textual summary"
                "of the provided output make sure to follow the guidelines provided.\n"
                "- Focus on the key information and maintain accuracy.\n"
                "- Make sure to not exceed the max limit of {max_length} characters.\n"
                "- Don't produce an overly long summary if it doesn't make sense.\n"
            ),
        ),
        HumanMessagePromptTemplate.from_template(
            "Please summarize the following output:\n{tool_output}\n",
        ),
    ])

    def __init__(self, llm: BaseChatModel, summary_max_length: int = 500) -> None:
        """Initialize the model."""
        self.llm = llm
        self.summary_max_length = summary_max_length

    def invoke(self, state: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state."""
        messages = state["messages"]
        last_message = messages[-1] if len(messages) > 0 else None
        if (
            not isinstance(last_message, ToolMessage)
            or not isinstance(last_message.artifact, Output)
        ):
            return {"messages": [last_message]}

        logger().debug(f"Invoke SummarizerModel on the tool output of {last_message.name}.")
        tool_output = last_message.content
        try:
            summary: BaseMessage = self.llm.invoke(
                self.summarizer_prompt.format_messages(
                    tool_output=tool_output,
                    max_length=self.summary_max_length,
                ),
            )
            last_message.artifact.summary = summary.content  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001 - we want to catch all exceptions
            logger().error("Error in SummarizerModel invoke (Skipping summaries): " + str(e))

        return {"messages": [last_message]}
