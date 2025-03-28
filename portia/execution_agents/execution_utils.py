"""Agent execution utilities.

This module contains utility functions for managing agent execution flow.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, cast

import instructor
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessagesState

from portia.clarification import Clarification
from portia.common import validate_extras_dependencies
from portia.errors import InvalidAgentOutputError, ToolFailedError, ToolRetryError
from portia.execution_agents.base_execution_agent import Output

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from openai.types.chat import ChatCompletionMessageParam
    from pydantic import BaseModel

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


def map_message_types_for_instructor(
    messages: list[BaseMessage],
) -> list[ChatCompletionMessageParam]:
    """Map the message types to the correct format for the LLM provider.

    Args:
        messages (list[BaseMessage]): Input Langchain messages

    Returns:
        list[dict]: The mapped messages.

    """
    def _map_message(message: BaseMessage) -> ChatCompletionMessageParam:
        match message.type:
            case "human":
                return {"role": "user", "content": message.text()}
            case "system":
                return {"role": "system", "content": message.text()}
            case _:
                raise ValueError(f"Unsupported message type: {message.type}")

    return [_map_message(message) for message in messages]


def invoke_structured_output(
    model: BaseChatModel,
    response_model: type[BaseModel],
    messages: list[BaseMessage],
) -> dict[Any, Any] | BaseModel:
    """Invoke a model with structured output.

    This function allows us to dispatch to the structured output method that works with
    the LLM provider.

    Args:
        model (BaseChatModel): The LangChain model to invoke.
        response_model (type[T]): The Pydantic model to use as schema for structured output.
        messages (list[BaseMessage]): The message input to the model.

    Returns:
        T: The deserialized Pydantic model from the LLM provider.

    """
    # We match on class name because not all these types are guaranteed to be installed
    match model.__class__.__name__:
        case "ChatOpenAI" | "ChatAnthropic" | "ChatMistralAI" | "AzureChatOpenAI":
            return model.with_structured_output(
                response_model,
                method="function_calling",
            ).invoke(messages)
        case "ChatGoogleGenerativeAI":
            validate_extras_dependencies("google")
            import google.generativeai as genai
            if TYPE_CHECKING:
                from langchain_google_genai import ChatGoogleGenerativeAI

            google_model = cast("ChatGoogleGenerativeAI", model)
            api_key = (
                google_model.google_api_key.get_secret_value()
                if google_model.google_api_key
                else None
            )
            genai.configure(api_key=api_key)  # pyright: ignore[reportPrivateImportUsage]
            client = instructor.from_gemini(
                client=genai.GenerativeModel(  # pyright: ignore[reportPrivateImportUsage]
                    model_name=google_model.model,
                ),
                mode=instructor.Mode.GEMINI_JSON,
            )
            return client.messages.create(  # pyright: ignore[reportReturnType]
                messages=map_message_types_for_instructor(messages),
                response_model=response_model,
            )
        case _:
            raise ValueError(f"Unsupported model type: {model.__class__.__name__}")
