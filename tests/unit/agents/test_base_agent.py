"""Test simple agent."""

from __future__ import annotations

from pydantic import SecretStr

from portia.agents.base_agent import BaseAgent
from portia.config import Config
from tests.utils import get_test_workflow


def test_base_agent_default_context() -> None:
    """Test default context."""
    plan, workflow = get_test_workflow()
    agent = BaseAgent(
        plan.steps[0],
        workflow,
        Config.from_default(
            openai_api_key=SecretStr("123"),
        ),
        None,
    )
    context = agent.get_system_context()
    assert context is not None
    assert "value: 1" in context


def test_base_agent_default_context_with_extensions() -> None:
    """Test default context."""
    plan, workflow = get_test_workflow()
    agent = BaseAgent(
        plan.steps[0],
        workflow,
        Config.from_default(
            agent_system_context_extension=["456"],
            openai_api_key=SecretStr("123"),
        ),
        None,
    )
    context = agent.get_system_context()
    assert context is not None
    assert "456" in context
