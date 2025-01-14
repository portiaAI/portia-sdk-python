"""Test simple agent."""

from __future__ import annotations

from portia.agents.base_agent import BaseAgent
from portia.context import execution_context
from tests.utils import get_test_config, get_test_workflow


def test_base_agent_default_context() -> None:
    """Test default context."""
    plan, workflow = get_test_workflow()
    agent = BaseAgent(
        plan.steps[0],
        workflow,
        get_test_config(),
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
        get_test_config(
            agent_system_context_extension=["456"],
        ),
        None,
    )
    with execution_context(agent_system_context_extension=["456"]):
        context = agent.get_system_context()
    assert context is not None
    assert "456" in context
