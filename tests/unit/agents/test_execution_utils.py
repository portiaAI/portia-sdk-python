"""Test execution utilities."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, MessagesState

from portia import tool_call
from portia.agents.base_agent import Output
from portia.agents.execution_utils import (
    MAX_RETRIES,
    AgentNode,
    next_state_after_tool_call,
    process_output,
    tool_call_or_end,
)
from portia.clarification import InputClarification
from portia.errors import InvalidAgentOutputError, ToolFailedError, ToolRetryError
from tests.utils import AdditionTool


def test_next_state_after_tool_call_no_error() -> None:
    """Test next state when tool call succeeds."""
    messages: list[ToolMessage] = [
        ToolMessage(
            content="Success message",
            tool_call_id="123",
            name="test_tool",
        ),
    ]
    state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

    result = next_state_after_tool_call(state)

    assert result == END


def test_next_state_after_tool_call_with_summarize() -> None:
    """Test next state when tool call succeeds and should summarize."""
    tool = AdditionTool()
    tool.should_summarize = True

    messages: list[ToolMessage] = [
        ToolMessage(
            content="Success message",
            tool_call_id="123",
            name="test_tool",
        ),
    ]
    state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

    result = next_state_after_tool_call(state, tool)

    assert result == AgentNode.SUMMARIZER


def test_next_state_after_tool_call_with_error_retry() -> None:
    """Test next state when tool call fails and max retries reached."""
    for i in range(1, MAX_RETRIES + 1):
        messages: list[ToolMessage] = [
            ToolMessage(
                content=f"ToolSoftError: Error {j}",
                tool_call_id=str(j),
                name="test_tool",
            )
            for j in range(1, i + 1)
        ]
        state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

        result = next_state_after_tool_call(state)

        expected_state = END if i == MAX_RETRIES else AgentNode.TOOL_AGENT
        assert result == expected_state, f"Failed at retry {i}"


def test_tool_call_or_end() -> None:
    """Test tool_call_or_end state transitions."""
    message_with_calls = AIMessage(content="test")
    state_with_calls: MessagesState = {"messages": [message_with_calls]}  # type: ignore  # noqa: PGH003

    message_without_calls = HumanMessage(content="test")
    state_without_calls: MessagesState = {"messages": [message_without_calls]}  # type: ignore  # noqa: PGH003

    assert tool_call_or_end(state_with_calls) == AgentNode.TOOLS
    assert tool_call_or_end(state_without_calls) == END


def test_process_output_with_clarifications() -> None:
    """Test process_output with clarifications."""
    clarifications = [InputClarification(argument_name="test", user_guidance="test")]
    message = HumanMessage(content="test")

    result = process_output(message, clarifications=clarifications)  # type: ignore  # noqa: PGH003

    assert isinstance(result, Output)
    assert result.value == clarifications


def test_process_output_with_tool_errors() -> None:
    """Test process_output with tool errors."""
    tool = AdditionTool()

    soft_error = ToolMessage(content="ToolSoftError: test", tool_call_id="1", name="test")
    hard_error = ToolMessage(content="ToolHardError: test", tool_call_id="1", name="test")

    with pytest.raises(ToolRetryError):
        process_output(soft_error, tool)

    with pytest.raises(ToolFailedError):
        process_output(hard_error, tool)


def test_process_output_with_invalid_message() -> None:
    """Test process_output with invalid message."""
    invalid_message = AIMessage(content="test")

    with pytest.raises(InvalidAgentOutputError):
        process_output(invalid_message)


def test_process_output_with_output_artifacts() -> None:
    """Test process_output with outpu artifacts."""
    message = ToolMessage(tool_call_id="1", content="", artifact=Output(value="test"))

    result = process_output(message, clarifications=[])

    assert isinstance(result, Output)
    assert result.value == "test"


def test_process_output_with_artifacts() -> None:
    """Test process_output with artifacts."""
    message = ToolMessage(tool_call_id="1", content="", artifact="test")

    result = process_output(message, clarifications=[])

    assert isinstance(result, Output)
    assert result.value == "test"


def test_process_output_with_content() -> None:
    """Test process_output with content."""
    message = ToolMessage(tool_call_id="1", content="test")

    result = process_output(message, clarifications=[])

    assert isinstance(result, Output)
    assert result.value == "test"


def test_process_output_with_human_message() -> None:
    """Test process_output with outpu artifacts."""
    message = HumanMessage(tool_call_id="1", content="test")

    result = process_output(message, clarifications=[])

    assert isinstance(result, Output)
    assert result.value == "test"
