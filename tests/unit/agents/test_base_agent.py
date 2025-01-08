"""Test simple agent."""

from __future__ import annotations

from portia.agents.base_agent import BaseAgent


def test_base_agent_default_context() -> None:
    """Test default context."""
    agent = BaseAgent(description="123", inputs=[])
    assert agent.system_context is not None


def test_base_agent_default_context_with_extensions() -> None:
    """Test default context."""
    agent = BaseAgent(description="123", inputs=[], system_context_extension=["456"])
    assert agent.system_context is not None
    assert "456" in agent.system_context
