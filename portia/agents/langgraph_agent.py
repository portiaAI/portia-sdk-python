from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage
from langgraph.graph import END, MessagesState
from portia.agents.base_agent import BaseAgent
from portia.config import Config
from portia.plan import Step
from portia.tool import Tool
from portia.workflow import Workflow

if TYPE_CHECKING:
    from langchain.tools import StructuredTool



class LanggraphAgent(BaseAgent):
    """Agent responsible for achieving a task by using langgraph."""

    MAX_RETRIES = 4

    def __init__(self, step: Step, workflow: Workflow, config: Config, tool: Tool | None = None):
        """Initialize the agent."""
        super().__init__(step, workflow, config, tool)

    def next_state_after_tool_call(self, state: MessagesState) -> Literal["tool_agent", "summarizer", END]:  # type: ignore  # noqa: PGH003
        """Determine the next state after the tool call. 
        If the tool has an error, we will retry the call until MAX_RETRIES.
        If the tool is configured to summarize, we will summarize the output.
        Otherwise, we will end the workflow.
        """
        messages = state["messages"]
        last_message = messages[-1]
        errors = [msg for msg in messages if "ToolSoftError" in msg.content]

        if "ToolSoftError" in last_message.content and len(errors) < LanggraphAgent.MAX_RETRIES:
            return "tool_agent"
        if (
            not "ToolSoftError" in last_message.content
            and self.tool
            and getattr(self.tool, "should_summarize", False)
        ):
            return "summarizer"
        # Otherwise, we return END as we either succeeded or failed the last retry.
        return END
