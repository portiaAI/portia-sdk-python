"""Tests for Calculator Tool."""

from unittest.mock import patch

import pytest

from portia.errors import ToolHardError
from portia.open_source_tools.calculator_tool import CalculatorTool
from tests.utils import get_test_tool_context


@pytest.fixture
def calculator_tool() -> CalculatorTool:
    """Return Calculator Tool."""
    return CalculatorTool()


def test_math_expression_conversion(calculator_tool: CalculatorTool) -> None:
    """Test expression conversion."""
    assert calculator_tool.math_expression("What is 3 added to 5?") == "3 + 5"
    assert calculator_tool.math_expression("12 multiplied by 4") == "12 * 4"
    assert calculator_tool.math_expression("20 divided by 5") == "20 / 5"
    assert calculator_tool.math_expression("divide 20 by 5") == "20 / 5"
    assert calculator_tool.math_expression("8 subtracted from 15") == "15 - 8"
    assert calculator_tool.math_expression("subtract 7 from 14") == "14 - 7"
    assert calculator_tool.math_expression("multiply 6 by 3") == "6 * 3"


def test_run_valid_expressions(calculator_tool: CalculatorTool) -> None:
    """Test valid expressions."""
    context = get_test_tool_context()
    assert calculator_tool.run(context, "3 plus 5") == 8.0
    assert calculator_tool.run(context, "10 divided by 2") == 5.0
    assert calculator_tool.run(context, "6 times 3") == 18.0
    assert calculator_tool.run(context, "15 minus 4") == 11.0
    assert calculator_tool.run(context, "212 + 14") == 226
    assert calculator_tool.run(context, "300 - 14") == 286
    assert calculator_tool.run(context, "3 x 2") == 6
    assert calculator_tool.run(context, "What is the sum of 17.42 and 16.72") == 34.15


def test_run_invalid_expressions(calculator_tool: CalculatorTool) -> None:
    """Test invalid expressions."""
    context = get_test_tool_context()
    with pytest.raises(ToolHardError):
        calculator_tool.run(context, "what is the meaning of life?")

    with pytest.raises(ToolHardError):
        calculator_tool.run(context, "")

    with pytest.raises(ToolHardError):
        calculator_tool.run(context, "")

    with patch.object(CalculatorTool, "math_expression", return_value=None):
        patched_tool = CalculatorTool()
        with pytest.raises(ToolHardError):
            patched_tool.run(context, " ")

    with pytest.raises(ToolHardError):
        calculator_tool.run(context, "5 + 3 * x")

    with pytest.raises(ToolHardError, match="Error evaluating expression."):
        calculator_tool.run(context, "subtract (def myclass) from 10")


def test_run_division_by_zero(calculator_tool: CalculatorTool) -> None:
    """Test divide by zero."""
    context = get_test_tool_context()
    with pytest.raises(ToolHardError, match="Error: Division by zero"):
        calculator_tool.run(context, "10 divided by 0")


def test_run_complex_expressions(calculator_tool: CalculatorTool) -> None:
    """Test complex."""
    context = get_test_tool_context()
    assert calculator_tool.run(context, "(3 plus 5) times 2") == 16.0
    assert calculator_tool.run(context, "(10 minus 3) divided by 2") == 3.5


def test_run_decimal_numbers(calculator_tool: CalculatorTool) -> None:
    """Test decimals."""
    context = get_test_tool_context()
    assert calculator_tool.run(context, "3.5 plus 2.5") == 6.0
    assert calculator_tool.run(context, "7.2 divided by 3.6") == 2.0
