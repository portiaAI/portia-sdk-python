"""Test toolless agent."""

from unittest import mock

import pytest

from portia.agents.toolless_agent import ToolLessAgent, ToolLessModel
from portia.config import Config
from portia.llm_wrapper import LLMWrapper


def test_toolless_agent_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running an agent without a tool."""
    mock_invoke = mock.Mock(return_value={"messages": ["invoked"]})
    monkeypatch.setattr(ToolLessModel, "invoke", mock_invoke)

    agent = ToolLessAgent(
        description="Write a sentence with every letter of the alphabet.",
        inputs=[],
        tool=None,
        clarifications=[],
        system_context=[],
    )

    output = agent.execute_sync(llm=LLMWrapper(Config()).to_langchain(), step_outputs={})
    assert mock_invoke.called
    assert output.value == "invoked"
