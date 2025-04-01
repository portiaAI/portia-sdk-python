"""Agent execution utilities.

This module contains utility functions for managing agent execution flow.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessagesState

from portia.clarification import Clarification
from portia.errors import InvalidAgentOutputError, ToolFailedError, ToolRetryError
from portia.execution_agents.output import Output

if TYPE_CHECKING:
    from portia.config import Config
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
    config: Config,
    state: MessagesState,
    tool: Tool | None = None,
) -> Literal[AgentNode.TOOL_AGENT, AgentNode.SUMMARIZER, END]:  # type: ignore  # noqa: PGH003
    """Determine the next state after a tool call.

    This function checks the state after a tool call to determine if the run
    should proceed to the tool agent again, to the summarizer, or end.

    Args:
        config (Config): The configuration for the run.
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
        and (
            getattr(tool, "should_summarize", False)
            # If the value is larger than the threshold value, always summarise them as they are
            # too big to store the full value locally
            or config.exceeds_output_threshold(last_message.content)
        )
        and isinstance(last_message, ToolMessage)
        and not is_clarification(last_message.artifact)
    ):
        return AgentNode.SUMMARIZER
    return END


def is_clarification(artifact: Any) -> bool:  # noqa: ANN401
    """Check if the artifact is a clarification or list of clarifications."""
    return isinstance(artifact, Clarification) or (
        isinstance(artifact, list)
        and len(artifact) > 0
        and all(isinstance(item, Clarification) for item in artifact)
    )


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


def process_output(  # noqa: C901
    messages: list[BaseMessage],
    tool: Tool | None = None,
    clarifications: list[Clarification] | None = None,
) -> Output:
    """Process the output of the agent.

    This function processes the agent's output based on the type of message received.
    It raises errors if the tool encounters issues and returns the appropriate output.

    Args:
        messages (list[BaseMessage}): The set of messages received from the agent's plan_run.
        tool (Tool | None): The tool associated with the agent, if any.
        clarifications (list[Clarification] | None): A list of clarifications, if any.

    Returns:
        Output: The processed output, which can be an error, tool output, or clarification.

    Raises:
        ToolRetryError: If there was a soft error with the tool and retries are allowed.
        ToolFailedError: If there was a hard error with the tool.
        InvalidAgentOutputError: If the output from the agent is invalid.

    """
    output_values: list[Output] = []
    for message in messages:
        if "ToolSoftError" in message.content and tool:
            raise ToolRetryError(tool.id, str(message.content))
        if "ToolHardError" in message.content and tool:
            raise ToolFailedError(tool.id, str(message.content))
        if clarifications and len(clarifications) > 0:
            return Output[list[Clarification]](
                value=clarifications,
            )
        if isinstance(message, ToolMessage):
            if message.artifact and isinstance(message.artifact, Output):
                output_values.append(message.artifact)
            elif message.artifact:
                output_values.append(Output(value=message.artifact))
            else:
                output_values.append(Output(value=message.content))

    if len(output_values) == 0:
        raise InvalidAgentOutputError(str([message.content for message in messages]))

    # if there's only one output return just the value
    if len(output_values) == 1:
        output = output_values[0]
        if output.summary is None:
            output.summary = output.serialize_value(output.value)
        return output

    values = []
    summaries = []

    for output in output_values:
        values.append(output.value)
        summaries.append(output.summary or output.serialize_value(output.value))

    return Output(value=values, summary=", ".join(summaries))
