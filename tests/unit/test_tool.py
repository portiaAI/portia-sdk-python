"""Tests for the Tool class."""

import pytest

from portia.context import get_execution_context
from portia.errors import InvalidToolDescriptionError, ToolSoftError
from tests.utils import AdditionTool, ClarificationTool, ErrorTool


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


def test_tool_initialization_long_description() -> None:
    """Test initialization of a Tool."""

    class FakeAdditionTool(AdditionTool):
        description: str = "this is a description" * 100

    with pytest.raises(InvalidToolDescriptionError):
        FakeAdditionTool()


def test_tool_to_langchain() -> None:
    """Test langchain rep of a Tool."""
    tool = AdditionTool()
    tool.to_langchain(return_artifact=False)


def test_run_method(add_tool: AdditionTool) -> None:
    """Test the run method of the AddTool."""
    a, b = 1, 2
    ctx = get_execution_context()
    result = add_tool.run(ctx, a, b)
    assert result == a + b


def test_run_method_with_uncaught_error() -> None:
    """Test the _run method wraps errors."""
    tool = ErrorTool()
    with pytest.raises(ToolSoftError):
        tool._run(  # noqa: SLF001
            error_str="this is an error",
            return_uncaught_error=True,
            return_soft_error=False,
        )
