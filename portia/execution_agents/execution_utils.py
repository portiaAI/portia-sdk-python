"""Agent execution utilities.

This module contains utility functions for managing agent execution flow.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, MessagesState

from portia.clarification import Clarification
from portia.errors import InvalidAgentOutputError, ToolFailedError, ToolRetryError
from portia.execution_agents.base_execution_agent import Output

if TYPE_CHECKING:
    from portia.tool import Tool


class AgentNode(str, Enum):
    """Nodes for agent execution.

    This enumeration defines the different types of nodes that can be encountered
    during the agent execution process.

    Attributes:
        TOOL_AGENT (str): A node representing the tool agent.
        SUMMARIZER (str): A node representing the summarizer.
        TOOLS (str): A node representing the tools.
        ARGUMENT_VERIFIER (str): A node representing the argument verifier.
        ARGUMENT_PARSER (str): A node representing the argument parser.

    """

    TOOL_AGENT = "tool_agent"
    SUMMARIZER = "summarizer"
    TOOLS = "tools"
    ARGUMENT_VERIFIER = "argument_verifier"
    ARGUMENT_PARSER = "argument_parser"


MAX_RETRIES = 4


def next_state_after_tool_call(
    state: MessagesState,
    tool: Tool | None = None,
) -> Literal[AgentNode.TOOL_AGENT, AgentNode.SUMMARIZER, END]:  # type: ignore  # noqa: PGH003
    """Determine the next state after a tool call.

    This function checks the state after a tool call to determine if the run
    should proceed to the tool agent again, to the summarizer, or end.

    Args:
        state (MessagesState): The current state of the messages.
        tool (Tool | None): The tool involved in the call, if any.

    Returns:
        Literal[AgentNode.TOOL_AGENT, AgentNode.SUMMARIZER, END]: The next state to transition to.

    Raises:
        ToolRetryError: If the tool has an error and the maximum retry limit has not been reached.

    """
    messages = state["messages"]
    last_message = messages[-1]
    errors = [msg for msg in messages if "ToolSoftError" in msg.content]

    if "ToolSoftError" in last_message.content and len(errors) < MAX_RETRIES:
        return AgentNode.TOOL_AGENT
    if (
        "ToolSoftError" not in last_message.content
        and tool
        and getattr(tool, "should_summarize", False)
    ):
        return AgentNode.SUMMARIZER
    return END


def tool_call_or_end(
    state: MessagesState,
) -> Literal[AgentNode.TOOLS, END]:  # type: ignore  # noqa: PGH003
    """Determine if tool execution should continue.

    This function checks if the current state indicates that the tool execution
    should continue, or if the run should end.

    Args:
        state (MessagesState): The current state of the messages.

    Returns:
        Literal[AgentNode.TOOLS, END]: The next state to transition to.

    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls"):
        return AgentNode.TOOLS
    return END


def process_output(
    last_message: BaseMessage,
    tool: Tool | None = None,
    clarifications: list[Clarification] | None = None,
) -> Output:
    """Process the output of the agent.

    This function processes the agent's output based on the type of message received.
    It raises errors if the tool encounters issues and returns the appropriate output.

    Args:
        last_message (BaseMessage): The last message received in the agent's plan_run.
        tool (Tool | None): The tool associated with the agent, if any.
        clarifications (list[Clarification] | None): A list of clarifications, if any.

    Returns:
        Output: The processed output, which can be an error, tool output, or clarification.

    Raises:
        ToolRetryError: If there was a soft error with the tool and retries are allowed.
        ToolFailedError: If there was a hard error with the tool.
        InvalidAgentOutputError: If the output from the agent is invalid.

    """
    if "ToolSoftError" in last_message.content and tool:
        raise ToolRetryError(tool.id, str(last_message.content))
    if "ToolHardError" in last_message.content and tool:
        raise ToolFailedError(tool.id, str(last_message.content))
    if clarifications and len(clarifications) > 0:
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
        if not tool_output.summary:
            tool_output.summary = tool_output.serialize_value(tool_output.value)
        return tool_output
    if isinstance(last_message, HumanMessage):
        return Output(value=last_message.content)
    raise InvalidAgentOutputError(str(last_message.content))
