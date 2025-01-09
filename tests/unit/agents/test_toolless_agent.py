"""Test toolless agent."""

from unittest import mock

import pytest

from portia.agents.toolless_agent import ToolLessAgent, ToolLessModel
from portia.config import Config
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
