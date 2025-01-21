"""LanggraphAgent is an agent that uses langgraph to achieve a task.

It is a wrapper around the BaseAgent class that provides common utilities for Langgraph agents.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langgraph.graph import END, MessagesState
from portia.agents.base_agent import Output
from portia.clarification import Clarification
from portia.errors import InvalidAgentOutputError, ToolFailedError, ToolRetryError
from portia.tool import Tool

class AgentNode(str, Enum):
    """States for agent execution."""
    TOOL_AGENT = "tool_agent"
    SUMMARIZER = "summarizer"
    TOOLS = "tools"
    ARGUMENT_VERIFIER = "argument_verifier"
    ARGUMENT_PARSER = "argument_parser"

class AgentExecutionManager:
    """Manages agent execution, including state transitions and output processing."""
    
    MAX_RETRIES = 4

    def __init__(self, tool: Tool | None = None) -> None:
        self.tool = tool

    def next_state_after_tool_call(
        self,
        state: MessagesState,
    ) -> Literal[AgentNode.TOOL_AGENT, AgentNode.SUMMARIZER, END]: # type: ignore  # noqa: PGH003
        """Determine the next state after a tool call.

        If the tool has an error, we will retry the call until MAX_RETRIES.
        If the tool is configured to summarize, we will summarize the output.
        Otherwise, we will end the workflow.
        """
        messages = state["messages"]
        last_message = messages[-1]
        errors = [msg for msg in messages if "ToolSoftError" in msg.content]

        if (
            "ToolSoftError" in last_message.content
            and len(errors) < AgentExecutionManager.MAX_RETRIES
        ):
            return AgentNode.TOOL_AGENT
        if (
            "ToolSoftError" not in last_message.content
            and self.tool
            and getattr(self.tool, "should_summarize", False)
        ):
            return AgentNode.SUMMARIZER
        return END

    def tool_call_or_end(self, state: MessagesState) -> Literal[AgentNode.TOOLS, END]: # type: ignore  # noqa: PGH003
        """Determine if tool execution should continue."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls"):
            return AgentNode.TOOLS
        return END

    def process_output(self, last_message: BaseMessage, clarifications: list[Clarification] = []) -> Output:
        """Process the output of the agent."""
        if "ToolSoftError" in last_message.content and self.tool:
            raise ToolRetryError(self.tool.name, str(last_message.content))
        if "ToolHardError" in last_message.content and self.tool:
            raise ToolFailedError(self.tool.name, str(last_message.content))
        if len(clarifications) > 0:
            return Output[list[Clarification]](
                value=clarifications,
            )
        if isinstance(last_message, ToolMessage):
            if last_message.artifact and isinstance(last_message.artifact, Output):
                tool_output = last_message.artifact
            elif last_message.artifact:
                tool_output = Output(value=last_message.artifact)
            else:
                tool_output = Output(value=last_message.content)
            return tool_output
        if isinstance(last_message, HumanMessage):
            return Output(value=last_message.content)
        raise InvalidAgentOutputError(str(last_message.content))
    