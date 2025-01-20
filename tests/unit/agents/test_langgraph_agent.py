"""Test langgraph agent."""
from __future__ import annotations

from langchain_core.messages import ToolMessage
from langgraph.graph import END, MessagesState

from portia.agents.langgraph_agent import LanggraphAgent
from portia.plan import Step
from tests.utils import AdditionTool, get_test_config, get_test_workflow


def test_next_state_after_tool_call_no_error() -> None:
    """Test next state when tool call succeeds."""
    # Arrange
    config = get_test_config()
    workflow = get_test_workflow()
    step = Step(task="test_task", output="test_output")
    agent = LanggraphAgent(step=step, workflow=workflow, config=config)

    messages: list[ToolMessage] = [
        ToolMessage(
            content="Success message",
            tool_call_id="123",
            name="test_tool",
        ),
    ]
    state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

    # Act
    result = agent.next_state_after_tool_call(state)

    # Assert
    assert result == END


def test_next_state_after_tool_call_with_summarize() -> None:
    """Test next state when tool call succeeds and should summarize."""
    # Arrange
    config = get_test_config()
    workflow = get_test_workflow()
    step = Step(task="test_task", output="test_output")
    tool = AdditionTool()
    tool.should_summarize = True
    agent = LanggraphAgent(step=step, workflow=workflow, config=config, tool=tool)

    messages: list[ToolMessage] = [
        ToolMessage(
            content="Success message",
            tool_call_id="123",
            name="test_tool",
        ),
    ]
    state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

    # Act
    result = agent.next_state_after_tool_call(state)

    # Assert
    assert result == "summarizer"


def test_next_state_after_tool_call_with_error_retry() -> None:
    """Test next state when tool call fails and max retries reached."""
    # Arrange
    config = get_test_config()
    workflow = get_test_workflow()
    step = Step(task="test_task", output="test_output")
    agent = LanggraphAgent(step=step, workflow=workflow, config=config)

    # Test each retry state
    for i in range(1, LanggraphAgent.MAX_RETRIES + 1):
        messages: list[ToolMessage] = [
            ToolMessage(
                content=f"ToolSoftError: Error {j}",
                tool_call_id=str(j),
                name="test_tool",
            )
            for j in range(1, i + 1)
        ]
        state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

        # Act
        result = agent.next_state_after_tool_call(state)

        # Assert
        expected_state = END if i == agent.MAX_RETRIES else "tool_agent"
        assert result == expected_state, f"Failed at retry {i}"
