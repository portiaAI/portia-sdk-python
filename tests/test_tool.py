"""Tests for the Tool class."""

import pytest

from tests.utils import AdditionTool


@pytest.fixture
def add_tool() -> AdditionTool:
    """Fixture to create a mock tool instance."""
    return AdditionTool()


def test_tool_initialization(add_tool: AdditionTool) -> None:
    """Test initialization of a Tool."""
    assert add_tool.name == "Add Tool"
    assert add_tool.id == "add_tool"
    assert add_tool.description == "Takes two numbers and adds them together"


def test_run_method(add_tool: AdditionTool) -> None:
    """Test the run method of the AddTool."""
    a, b = 1, 2
    result = add_tool.run(a, b)
    assert result == a + b
