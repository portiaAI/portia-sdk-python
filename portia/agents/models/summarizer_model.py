"""Summarizer model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from portia.agents.base_agent import Output
from portia.logger import logger

if TYPE_CHECKING:
    from langchain.chat_models.base import BaseChatModel
    from langgraph.graph import MessagesState


class SummarizerOutput(BaseModel):
    """Model for the structured summary output."""

    short_summary: str = Field(
        description="Brief one-line summary of the output",
    )
    long_summary: str = Field(
        description="Detailed summary of the output",
    )

class SummarizerModel:
    """Model to generate two summaries (short and detailed) for the tool output.

    This model is used only on the tool output message.
    """

    summarizer_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=(
                "You are a highly skilled summarizer. Your task is to create two summaries"
                "of the provided output\n"
                "1. short_summary: A brief, one-line summary "
                "for the output (max limit 100 characters)\n"
                "2. long_summary: A detailed summary for the output "
                "(max limit 500 characters)\n"
                "Focus on the key information and maintain accuracy. "
                "MAKE SURE to not exceed the max limit provided for each summary."
            ),
        ),
        HumanMessagePromptTemplate.from_template(
            "Please summarize the following output:\n{tool_output}\n",
        ),
    ])

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the model."""
        self.llm = llm

    def invoke(self, state: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state."""
        messages = state["messages"]
        last_message = messages[-1] if len(messages) > 0 else None
        if (
            not isinstance(last_message, ToolMessage)
            or not isinstance(last_message.artifact, Output)
        ):
            return {"messages": [last_message]}

        logger.info(f"Invoke SummarizerModel on the tool output of {last_message.name}.")
        tool_output = last_message.content
        model = self.llm.with_structured_output(SummarizerOutput)
        try:
            summary = model.invoke(
                self.summarizer_prompt.format_messages(tool_output=tool_output),
            )
            summary = SummarizerOutput.model_validate(summary)
            last_message.artifact.short_summary = summary.short_summary
            last_message.artifact.long_summary = summary.long_summary
        except Exception as e:  # noqa: BLE001 - we want to catch all exceptions
            logger.error("Error in SummarizerModel invoke (Skipping summaries): " + str(e))

        return {"messages": [last_message]}
