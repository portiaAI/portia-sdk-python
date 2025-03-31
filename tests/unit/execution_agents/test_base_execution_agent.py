"""Test simple agent."""

from __future__ import annotations

from portia.execution_agents.base_execution_agent import BaseExecutionAgent
from portia.execution_context import execution_context
from tests.utils import get_test_config, get_test_plan_run


def test_base_agent_default_context() -> None:
    """Test default context."""
    plan, plan_run = get_test_plan_run()
    agent = BaseExecutionAgent(
        plan.steps[0],
        plan_run,
        get_test_config(),
        None,
    )
    context = agent.get_system_context()
    assert context is not None
    assert "value: 1" in context


def test_base_agent_default_context_with_extensions() -> None:
    """Test default context."""
    plan, plan_run = get_test_plan_run()
    agent = BaseExecutionAgent(
        plan.steps[0],
        plan_run,
        get_test_config(),
        None,
    )
    with execution_context(agent_system_context_extension=["456"]):
        context = agent.get_system_context()
    assert context is not None
    assert "456" in context
