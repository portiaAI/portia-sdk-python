"""Tests for the Tool class."""

import pytest
from pydantic import HttpUrl

from portia.clarification import ActionClarification
from tests.utils import AdditionTool, ClarificationTool


@pytest.fixture
def add_tool() -> AdditionTool:
    """Fixture to create a mock tool instance."""
    return AdditionTool()


@pytest.fixture
def clarification_tool() -> ClarificationTool:
    """Fixture to create a mock tool instance."""
    return ClarificationTool()


def test_tool_initialization(add_tool: AdditionTool) -> None:
    """Test initialization of a Tool."""
    assert add_tool.name == "Add Tool"
    assert add_tool.description == "Takes two numbers and adds them together"


def test_run_method(add_tool: AdditionTool) -> None:
    """Test the run method of the AddTool."""
    a, b = 1, 2
    result = add_tool.run(a, b)
    assert result == a + b
