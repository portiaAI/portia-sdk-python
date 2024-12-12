"""Tests for the Tool class."""

import pytest

from portia.tool import Tool


class AddTool(Tool[int]):
    """Adds two numbers together."""

    id: str = "add_tool"
    name: str = "Add Tool"
    description: str = "Takes two numbers and adds them together"

    def run(self, a: int, b: int) -> int:
        """Return sum(a, b)."""
        return a + b


@pytest.fixture
def add_tool() -> AddTool:
    """Fixture to create a mock tool instance."""
    return AddTool()


def test_tool_initialization(add_tool: AddTool) -> None:
    """Test initialization of a Tool."""
    assert add_tool.name == "Add Tool"
    assert add_tool.id == "add_tool"
    assert add_tool.description == "Takes two numbers and adds them together"


def test_run_method(add_tool: AddTool) -> None:
    """Test the run method of the AddTool."""
    a, b = 1, 2
    result = add_tool.run(a, b)
    assert result == a + b
