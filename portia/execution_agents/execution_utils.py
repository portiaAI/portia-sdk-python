"""Agent execution utilities.

This module contains utility functions for managing agent execution flow.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from jinja2 import Template
from langchain_core.messages import ToolMessage
from langgraph.graph import END, MessagesState
from pydantic import ValidationError

from portia.clarification import Clarification, InputClarification
from portia.errors import (
    InvalidAgentOutputError,
    InvalidPlanRunStateError,
    ToolFailedError,
    ToolRetryError,
)
from portia.execution_agents.output import LocalDataValue, Output

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from portia.execution_agents.context import StepInput
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
        MEMORY_EXTRACTION (str): A node representing the memory extraction step.

    """

    TOOL_AGENT = "tool_agent"
    SUMMARIZER = "summarizer"
    TOOLS = "tools"
    ARGUMENT_VERIFIER = "argument_verifier"
    ARGUMENT_PARSER = "argument_parser"
    MEMORY_EXTRACTION = "memory_extraction"


MAX_RETRIES = 4


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
    messages = state["messages"]
    if len(messages) > 0 and hasattr(messages[-1], "tool_calls"):
        return AgentNode.TOOLS
    return END


def get_arg_value_with_templating(step_inputs: list[StepInput], arg: Any) -> Any:  # noqa: ANN401
    """Return the value of an argument, handling any templating required."""
    # Directly apply templating in strings
    if isinstance(arg, str):
        if any(
            # Allow with or without spaces
            f"{{{{{step_input.name}}}}}" in arg or f"{{{{ {step_input.name} }}}}" in arg
            for step_input in step_inputs
        ):
            return _template_inputs_into_arg_value(arg, step_inputs)
        return arg

    # Recursively handle lists and dicts
    if isinstance(arg, list):
        return [get_arg_value_with_templating(step_inputs, item) for item in arg]
    if isinstance(arg, dict):
        return {k: get_arg_value_with_templating(step_inputs, v) for k, v in arg.items()}

    # We don't yet support templating for other types
    return arg


def _template_inputs_into_arg_value(arg_value: str, step_inputs: list[StepInput]) -> str:
    """Template inputs into an argument value."""
    template_args = {}
    for step_input in step_inputs:
        input_name = step_input.name

        # jinja can't handle inputs that start with $, so remove any leading $
        # this also handles the case where the parser accidentlly misses off the $ in templating,
        # which does happen (e.g. it uses {{input_name}} instead of {{$input_name}}
        input_name = input_name.lstrip("$")
        arg_value = arg_value.replace(step_input.name, input_name)
        template_args[input_name] = step_input.value

    return Template(arg_value).render(**template_args)


def template_in_required_inputs(
    response: BaseMessage,
    step_inputs: list[StepInput],
) -> BaseMessage:
    """Template any required inputs into the tool calls."""
    for tool_call in response.tool_calls:  # pyright: ignore[reportAttributeAccessIssue]
        if not isinstance(tool_call.get("args"), dict):
            raise InvalidPlanRunStateError("Tool call missing args field")

        for arg_name, arg_value in tool_call.get("args").items():
            tool_call["args"][arg_name] = get_arg_value_with_templating(step_inputs, arg_value)

    return response


def process_output(  # noqa: C901
    messages: list[BaseMessage],
    tool: Tool | None = None,
    clarifications: list[Clarification] | None = None,
) -> Output:
    """Process the output of the agent.

    This function processes the agent's output based on the type of message received.
    It raises errors if the tool encounters issues and returns the appropriate output.

    Args:
        messages (list[BaseMessage]): The set of messages received from the agent's plan_run.
        tool (Tool | None): The tool associated with the agent, if any.
        clarifications (list[Clarification] | None): A list of clarifications, if any.

    Returns:
        Output: The processed output, which can be an error, tool output, or clarification.

    Raises:
        ToolRetryError: If there was a soft error with the tool and retries are allowed.
        ToolFailedError: If there was a hard error with the tool.
        InvalidAgentOutputError: If the output from the agent is invalid.

    """
    if clarifications and len(clarifications) > 0:
        return LocalDataValue(value=clarifications)

    output_values: list[Output] = []
    for message in messages:
        if "ToolSoftError" in message.content and tool:
            raise ToolRetryError(tool.id, str(message.content))
        if "ToolHardError" in message.content and tool:
            raise ToolFailedError(tool.id, str(message.content))
        if isinstance(message, ToolMessage):
            try:
                clarification = InputClarification.model_validate_json(message.content)  # pyright: ignore[reportArgumentType]
                return LocalDataValue(value=[clarification])
            except ValidationError:
                pass

            if message.artifact and isinstance(message.artifact, Output):
                output_values.append(message.artifact)
            elif message.artifact:
                output_values.append(LocalDataValue(value=message.artifact))
            else:
                output_values.append(LocalDataValue(value=message.content))

    if len(output_values) == 0:
        raise InvalidAgentOutputError(str([message.content for message in messages]))

    # if there's only one output return just the value
    if len(output_values) == 1:
        output = output_values[0]
        return LocalDataValue(
            value=output.get_value(),
            summary=output.get_summary() or output.serialize_value(),
        )

    values = []
    summaries = []

    for output in output_values:
        values.append(output.get_value())
        summaries.append(output.get_summary() or output.serialize_value())

    return LocalDataValue(value=values, summary=", ".join(summaries))
