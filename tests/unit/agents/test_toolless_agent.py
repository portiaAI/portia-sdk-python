"""Test toolless agent."""

from unittest import mock

import pytest
from langgraph.graph import MessagesState

from portia.agents.toolless_agent import ToolLessAgent, ToolLessModel
from portia.plan import Step
from tests.utils import get_test_config, get_test_workflow


def test_toolless_agent_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running an agent without a tool."""
    mock_invoke = mock.Mock(return_value={"messages": ["invoked"]})
    monkeypatch.setattr(ToolLessModel, "invoke", mock_invoke)

    (plan, workflow) = get_test_workflow()
    agent = ToolLessAgent(step=plan.steps[0], workflow=workflow, config=get_test_config())

    output = agent.execute_sync()
    assert mock_invoke.called
    assert output.value == "invoked"


def test_toolless_agent_regular_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running a regular task uses the system context."""
    # Mock the invoke method to capture the context
    captured_context = None
    def mock_invoke(self: ToolLessModel, _: MessagesState) -> dict[str, list[str]]:
        nonlocal captured_context
        captured_context = self.context
        return {"messages": ["Regular response"]}

    monkeypatch.setattr(ToolLessModel, "invoke", mock_invoke)

    # Create a plan with a regular step
    (plan, workflow) = get_test_workflow()
    plan.steps = [
        Step(
            task="Get weather in London",
            output="$london_weather",
        ),
    ]

    # Create agent for a regular step
    agent = ToolLessAgent(
        step=plan.steps[0],
        workflow=workflow,
        config=get_test_config(),
    )

    # Execute the agent
    output = agent.execute_sync()

    assert captured_context is not None
    assert "System Context:" in captured_context
    assert output.value == "Regular response"
