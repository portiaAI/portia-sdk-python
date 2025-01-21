"""Test agent execution manager."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, MessagesState

from portia.agents.agent_execution_manager import AgentExecutionManager, AgentNode
from portia.agents.base_agent import Output
from portia.clarification import InputClarification
from portia.errors import InvalidAgentOutputError, ToolFailedError, ToolRetryError
from tests.utils import AdditionTool


def test_next_state_after_tool_call_no_error() -> None:
    """Test next state when tool call succeeds."""
    manager = AgentExecutionManager()
    messages: list[ToolMessage] = [
        ToolMessage(
            content="Success message",
            tool_call_id="123",
            name="test_tool",
        ),
    ]
    state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

    result = manager.next_state_after_tool_call(state)

    assert result == END


def test_next_state_after_tool_call_with_summarize() -> None:
    """Test next state when tool call succeeds and should summarize."""
    tool = AdditionTool()
    tool.should_summarize = True
    manager = AgentExecutionManager(tool)

    messages: list[ToolMessage] = [
        ToolMessage(
            content="Success message",
            tool_call_id="123",
            name="test_tool",
        ),
    ]
    state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

    result = manager.next_state_after_tool_call(state)

    assert result == AgentNode.SUMMARIZER


def test_next_state_after_tool_call_with_error_retry() -> None:
    """Test next state when tool call fails and max retries reached."""
    manager = AgentExecutionManager()

    for i in range(1, AgentExecutionManager.MAX_RETRIES + 1):
        messages: list[ToolMessage] = [
            ToolMessage(
                content=f"ToolSoftError: Error {j}",
                tool_call_id=str(j),
                name="test_tool",
            )
            for j in range(1, i + 1)
        ]
        state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

        result = manager.next_state_after_tool_call(state)

        expected_state = END if i == AgentExecutionManager.MAX_RETRIES else AgentNode.TOOL_AGENT
        assert result == expected_state, f"Failed at retry {i}"


def test_should_continue_tool_execution() -> None:
    """Test should_continue_tool_execution state transitions."""
    manager = AgentExecutionManager()
    
    message_with_calls = AIMessage(content="test")
    state_with_calls: MessagesState = {"messages": [message_with_calls]}  # type: ignore
    
    message_without_calls = HumanMessage(content="test")
    state_without_calls: MessagesState = {"messages": [message_without_calls]}  # type: ignore

    assert manager.tool_call_or_end(state_with_calls) == AgentNode.TOOLS
    assert manager.tool_call_or_end(state_without_calls) == END


def test_process_output_with_clarifications() -> None:
    """Test process_output with clarifications."""
    manager = AgentExecutionManager()
    clarifications = [InputClarification(argument_name="test", user_guidance="test")]
    message = HumanMessage(content="test")

    result = manager.process_output(message, clarifications)

    assert isinstance(result, Output)
    assert result.value == clarifications


def test_process_output_with_tool_errors() -> None:
    """Test process_output with tool errors."""
    tool = AdditionTool()
    manager = AgentExecutionManager(tool)
    
    soft_error = ToolMessage(content="ToolSoftError: test", tool_call_id="1", name="test")
    hard_error = ToolMessage(content="ToolHardError: test", tool_call_id="1", name="test")

    try:
        manager.process_output(soft_error)
        assert False, "Should have raised ToolRetryError"
    except ToolRetryError:
        pass

    try:
        manager.process_output(hard_error)
        assert False, "Should have raised ToolFailedError"
    except ToolFailedError:
        pass


def test_process_output_with_invalid_message() -> None:
    """Test process_output with invalid message."""
    manager = AgentExecutionManager()
    invalid_message = AIMessage(content="test")

    try:
        manager.process_output(invalid_message)
        assert False, "Should have raised InvalidAgentOutputError"
    except InvalidAgentOutputError:
        pass 