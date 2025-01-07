"""Test simple agent."""

from __future__ import annotations

import asyncio

import pytest

from portia.agents.base_agent import BaseAgent, RequestClarificationTool
from portia.clarification import InputClarification


def test_request_clarification_tool() -> None:
    """Test request clarification tool."""
    tool = RequestClarificationTool()
    output = tool.run(tool_input={"missing_args": ["arg1"]})
    assert isinstance(output, list)
    assert isinstance(output[0], InputClarification)

    with pytest.raises(NotImplementedError):
        asyncio.run(tool.arun({"missing_args": ["arg1"]}))


def test_base_agent_default_context() -> None:
    """Test default context."""
    agent = BaseAgent(description="123", inputs=[])
    assert agent.system_context is not None


def test_base_agent_default_context_with_extensions() -> None:
    """Test default context."""
    agent = BaseAgent(description="123", inputs=[], system_context_extension=["456"])
    assert agent.system_context is not None
    assert "456" in agent.system_context
