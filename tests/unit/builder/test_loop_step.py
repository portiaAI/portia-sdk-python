"""Test the loop step."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from portia.builder.loop_step import LoopStep
from portia.builder.loops import LoopStepType, LoopType
from portia.builder.reference import Input, StepOutput
from portia.run_context import StepOutputValue


@pytest.fixture
def mock_run_data() -> Mock:
    """Create mock run data for testing LoopStep."""
    mock_data = Mock()
    mock_data.config = Mock()
    mock_data.storage = Mock()
    mock_data.step_output_values = []
    mock_data.plan = Mock()
    mock_data.plan_run = Mock()
    return mock_data


@pytest.fixture
def mock_plan_v2() -> Mock:
    """Create mock Plan for testing LoopStep."""
    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "loop_output"
    return mock_plan


def test_loop_step_initialization_with_condition() -> None:
    """Test LoopStep initialization with condition."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.DO_WHILE,
        loop_step_type=LoopStepType.END,
        args={"x": 10},
    )

    assert step.step_name == "test_loop"
    assert step.condition is not None
    assert step.loop_type == LoopType.DO_WHILE
    assert step.loop_step_type == LoopStepType.END
    assert step.args == {"x": 10}
    assert step.over is None


def test_loop_step_initialization_with_over() -> None:
    """Test LoopStep initialization with over reference."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.START,
        index=0,
    )

    assert step.step_name == "test_loop"
    assert step.over is not None
    assert step.loop_type == LoopType.FOR_EACH
    assert step.loop_step_type == LoopStepType.START
    assert step.index == 0
    assert step.condition is None


def test_loop_step_validation_error_both_none() -> None:
    """Test LoopStep validation error when both condition and over are None."""
    with pytest.raises(ValueError, match="Condition and over cannot both be None"):
        LoopStep(
            step_name="test_loop",
            condition=None,
            over=None,
            loop_type=LoopType.DO_WHILE,
            loop_step_type=LoopStepType.END,
        )


def test_loop_step_validation_success() -> None:
    """Test LoopStep validation success with valid parameters."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.DO_WHILE,
        loop_step_type=LoopStepType.END,
    )

    assert step.step_name == "test_loop"
    assert step.condition is not None


def test_current_loop_variable_with_over() -> None:
    """Test current_loop_variable method when over is set."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.START,
        index=1,
    )

    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value=["a", "b", "c"],
            description="s0",
            step_name="test_step",
            step_num=0,
        )
    ]
    mock_run_data.plan.steps = []

    result = step._current_loop_variable(mock_run_data)
    assert result == "b"


def test_current_loop_variable_with_over_none() -> None:
    """Test current_loop_variable method when over is None."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.DO_WHILE,
        loop_step_type=LoopStepType.END,
    )

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)
    assert result is None


def test_current_loop_variable_with_non_sequence() -> None:
    """Test current_loop_variable method with non-sequence value."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.START,
        index=0,
    )

    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value=42,  # Use an integer which is not indexable
            description="s0",
            step_name="test_step",
            step_num=0,
        )
    ]
    mock_run_data.plan.steps = []

    with pytest.raises(TypeError, match="Loop variable is not indexable"):
        step._current_loop_variable(mock_run_data)


def test_current_loop_variable_index_out_of_range() -> None:
    """Test current_loop_variable method with index out of range."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.START,
        index=5,
    )

    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value=["a", "b", "c"],
            description="s0",
            step_name="test_step",
            step_num=0,
        )
    ]
    mock_run_data.plan.steps = []

    result = step._current_loop_variable(mock_run_data)
    assert result is None


@pytest.mark.asyncio
async def test_loop_step_run_conditional_end_with_callable() -> None:
    """Test LoopStep run method for do-while end loop with callable condition."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 5,
        loop_type=LoopType.DO_WHILE,
        loop_step_type=LoopStepType.END,
        args={"x": 10},
    )

    mock_run_data = Mock()
    result = await step.run(run_data=mock_run_data)

    assert result.loop_result is True


@pytest.mark.asyncio
async def test_loop_step_run_conditional_end_with_string() -> None:
    """Test LoopStep run method for do-while end loop with string condition."""
    step = LoopStep(
        step_name="test_loop",
        condition="x > 5",
        loop_type=LoopType.DO_WHILE,
        loop_step_type=LoopStepType.END,
        args={"x": 10},
    )

    mock_run_data = Mock()
    mock_run_data.config = Mock()

    with patch("portia.builder.loop_step.ConditionalEvaluationAgent") as mock_agent_class:
        mock_agent = Mock()
        mock_agent.execute = AsyncMock(return_value=True)
        mock_agent_class.return_value = mock_agent

        result = await step.run(run_data=mock_run_data)

        assert result.loop_result is True


@pytest.mark.asyncio
async def test_loop_step_run_conditional_end_missing_condition() -> None:
    """Test LoopStep run method for do-while end loop with missing condition."""
    # Create a valid LoopStep first, then test the run method behavior
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.DO_WHILE,
        loop_step_type=LoopStepType.END,
    )

    mock_run_data = Mock()

    # Manually set condition to None to test the run method behavior
    step.condition = None
    with pytest.raises(ValueError, match="Condition is required for loop step"):
        await step.run(run_data=mock_run_data)


@pytest.mark.asyncio
async def test_loop_step_run_for_each_start() -> None:
    """Test LoopStep run method for for-each start loop."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.START,
        index=0,
    )

    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value=["a", "b", "c"],
            description="s0",
            step_name="test_step",
            step_num=0,
        )
    ]
    mock_run_data.plan.steps = []

    result = await step.run(run_data=mock_run_data)

    assert result.loop_result is True
    assert result.value == "a"
    assert step.index == 1  # Should be incremented


@pytest.mark.asyncio
async def test_loop_step_run_for_each_start_missing_over() -> None:
    """Test LoopStep run method for for-each start loop with missing over."""
    # Create a valid LoopStep first, then test the run method behavior
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.START,
    )

    mock_run_data = Mock()

    # Manually set over to None to test the run method behavior
    step.over = None
    with pytest.raises(ValueError, match="Over is required for for-each loop"):
        await step.run(run_data=mock_run_data)


@pytest.mark.asyncio
async def test_loop_step_run_for_each_start_no_value() -> None:
    """Test LoopStep run method for for-each start loop with no value."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.START,
        index=5,  # Out of range
    )

    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value=["a", "b", "c"],
            description="s0",
            step_name="test_step",
            step_num=0,
        )
    ]
    mock_run_data.plan.steps = []

    result = await step.run(run_data=mock_run_data)

    assert result.loop_result is False
    assert result.value is None


@pytest.mark.asyncio
async def test_loop_step_run_default_case() -> None:
    """Test LoopStep run method for while start case."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.WHILE,
        loop_step_type=LoopStepType.START,
        args={"x": 10},  # Provide the required argument
    )

    mock_run_data = Mock()
    result = await step.run(run_data=mock_run_data)

    assert result.loop_result is True


def test_loop_step_to_legacy_step_with_callable_condition() -> None:
    """Test LoopStep to_legacy_step method with callable condition."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.DO_WHILE,
        loop_step_type=LoopStepType.END,
        args={"x": 10},
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "loop_output"

    result = step.to_legacy_step(mock_plan)

    assert result.task == "Loop clause: If result of <lambda> is true"
    assert result.output == "loop_output"
    assert result.tool_id is None


def test_loop_step_to_legacy_step_with_string_condition() -> None:
    """Test LoopStep to_legacy_step method with string condition."""
    step = LoopStep(
        step_name="test_loop",
        condition="x > 0",
        loop_type=LoopType.DO_WHILE,
        loop_step_type=LoopStepType.END,
        args={"x": 10},
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "loop_output"

    result = step.to_legacy_step(mock_plan)

    assert result.task == "Loop clause: x > 0"
    assert result.output == "loop_output"
    assert result.tool_id is None


def test_loop_step_to_legacy_step_with_reference_args() -> None:
    """Test LoopStep to_legacy_step method with reference arguments."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.DO_WHILE,
        loop_step_type=LoopStepType.END,
        args={"x": StepOutput(0), "y": Input("test_input")},
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "loop_output"

    result = step.to_legacy_step(mock_plan)

    assert result.task == "Loop clause: If result of <lambda> is true"
    assert result.output == "loop_output"
    assert result.tool_id is None


# New tests for the updated interface


def test_loop_step_validation_error_condition_with_for_each() -> None:
    """Test LoopStep validation error when condition is set for for-each loop."""
    with pytest.raises(ValueError, match="Condition cannot be set for for-each loop"):
        LoopStep(
            step_name="test_loop",
            condition=lambda x: x > 0,
            over=StepOutput(0),
            loop_type=LoopType.FOR_EACH,
            loop_step_type=LoopStepType.START,
        )


def test_loop_step_validation_error_condition_and_over_both_set() -> None:
    """Test LoopStep validation error when both condition and over are set."""
    with pytest.raises(ValueError, match="Condition and over cannot both be set"):
        LoopStep(
            step_name="test_loop",
            condition=lambda x: x > 0,
            over=StepOutput(0),
            loop_type=LoopType.DO_WHILE,
            loop_step_type=LoopStepType.END,
        )


def test_loop_step_validation_error_over_with_while() -> None:
    """Test LoopStep validation error when over is set for while loop."""
    with pytest.raises(ValueError, match="Over cannot be set for while or do-while loop"):
        LoopStep(
            step_name="test_loop",
            condition=None,
            over=StepOutput(0),
            loop_type=LoopType.WHILE,
            loop_step_type=LoopStepType.START,
        )


def test_loop_step_validation_error_over_with_do_while() -> None:
    """Test LoopStep validation error when over is set for do-while loop."""
    with pytest.raises(ValueError, match="Over cannot be set for while or do-while loop"):
        LoopStep(
            step_name="test_loop",
            condition=None,
            over=StepOutput(0),
            loop_type=LoopType.DO_WHILE,
            loop_step_type=LoopStepType.END,
        )


@pytest.mark.asyncio
async def test_loop_step_run_while_start_with_callable() -> None:
    """Test LoopStep run method for while start loop with callable condition."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 5,
        loop_type=LoopType.WHILE,
        loop_step_type=LoopStepType.START,
        args={"x": 10},
    )

    mock_run_data = Mock()
    result = await step.run(run_data=mock_run_data)

    assert result.loop_result is True


@pytest.mark.asyncio
async def test_loop_step_run_while_start_with_string() -> None:
    """Test LoopStep run method for while start loop with string condition."""
    step = LoopStep(
        step_name="test_loop",
        condition="x > 5",
        loop_type=LoopType.WHILE,
        loop_step_type=LoopStepType.START,
        args={"x": 10},
    )

    mock_run_data = Mock()
    mock_run_data.config = Mock()

    with patch("portia.builder.loop_step.ConditionalEvaluationAgent") as mock_agent_class:
        mock_agent = Mock()
        mock_agent.execute = AsyncMock(return_value=True)
        mock_agent_class.return_value = mock_agent

        result = await step.run(run_data=mock_run_data)

        assert result.loop_result is True


@pytest.mark.asyncio
async def test_loop_step_run_do_while_end_with_callable() -> None:
    """Test LoopStep run method for do-while end loop with callable condition."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 5,
        loop_type=LoopType.DO_WHILE,
        loop_step_type=LoopStepType.END,
        args={"x": 10},
    )

    mock_run_data = Mock()
    result = await step.run(run_data=mock_run_data)

    assert result.loop_result is True


@pytest.mark.asyncio
async def test_loop_step_run_do_while_end_with_string() -> None:
    """Test LoopStep run method for do-while end loop with string condition."""
    step = LoopStep(
        step_name="test_loop",
        condition="x > 5",
        loop_type=LoopType.DO_WHILE,
        loop_step_type=LoopStepType.END,
        args={"x": 10},
    )

    mock_run_data = Mock()
    mock_run_data.config = Mock()

    with patch("portia.builder.loop_step.ConditionalEvaluationAgent") as mock_agent_class:
        mock_agent = Mock()
        mock_agent.execute = AsyncMock(return_value=True)
        mock_agent_class.return_value = mock_agent

        result = await step.run(run_data=mock_run_data)

        assert result.loop_result is True


@pytest.mark.asyncio
async def test_loop_step_run_while_start_missing_condition() -> None:
    """Test LoopStep run method for while start loop with missing condition."""
    # Create a valid LoopStep first, then test the run method behavior
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.WHILE,
        loop_step_type=LoopStepType.START,
    )

    mock_run_data = Mock()

    # Manually set condition to None to test the run method behavior
    step.condition = None
    with pytest.raises(ValueError, match="Condition is required for loop step"):
        await step.run(run_data=mock_run_data)


@pytest.mark.asyncio
async def test_loop_step_run_for_each_end() -> None:
    """Test LoopStep run method for for-each end loop."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.END,
        index=1,
    )

    mock_run_data = Mock()
    result = await step.run(run_data=mock_run_data)

    assert result.loop_result is True
    assert result.value is True


def test_current_loop_variable_with_none_over() -> None:
    """Test _current_loop_variable when over is None."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("test_step"),
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.END,
        index=0,
    )

    # Manually set over to None to test the method behavior
    step.over = None

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)

    assert result is None


def test_current_loop_variable_with_sequence_over() -> None:
    """Test _current_loop_variable when over is a Sequence."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=[["item1", "item2"], ["item3", "item4"]],
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.END,
        index=0,
    )

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)

    assert result == "item1"


def test_current_loop_variable_with_sequence_over_different_index() -> None:
    """Test _current_loop_variable with different index values."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=[["item1", "item2"], ["item3", "item4"]],
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.END,
        index=1,
    )

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)

    assert result == "item4"


def test_current_loop_variable_with_reference_over() -> None:
    """Test _current_loop_variable when over is a Reference."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("previous_step"),
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.END,
        index=0,
    )

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_name="previous_step", step_num=0, value=["ref_item1", "ref_item2"])
    ]

    # Mock the _resolve_references method to return the expected sequence
    with patch.object(step, "_resolve_references", return_value=["ref_item1", "ref_item2"]):
        result = step._current_loop_variable(mock_run_data)

    assert result == "ref_item1"


def test_current_loop_variable_with_reference_over_different_index() -> None:
    """Test _current_loop_variable with Reference over and different index."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("previous_step"),
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.END,
        index=1,
    )

    mock_run_data = Mock()

    # Mock the _resolve_references method to return the expected sequence
    with patch.object(step, "_resolve_references", return_value=["ref_item1", "ref_item2"]):
        result = step._current_loop_variable(mock_run_data)

    assert result == "ref_item2"


def test_current_loop_variable_with_non_sequence_resolved_value() -> None:
    """Test _current_loop_variable when resolved value is not a sequence."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("previous_step"),
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.END,
        index=0,
    )

    mock_run_data = Mock()

    # Mock the _resolve_references method to return a non-sequence value (integer)
    with (
        patch.object(step, "_resolve_references", return_value=42),
        pytest.raises(TypeError, match="Loop variable is not indexable"),
    ):
        step._current_loop_variable(mock_run_data)


def test_current_loop_variable_with_index_out_of_bounds() -> None:
    """Test _current_loop_variable when index is out of bounds."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=[["item1", "item2"], ["item3", "item4"], ["item5", "item6"]],
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.END,
        index=2,  # Valid index for outer sequence
    )

    mock_run_data = Mock()
    # This will access over[2] = ["item5", "item6"], then ["item5", "item6"][2]
    # which is out of bounds
    result = step._current_loop_variable(mock_run_data)

    assert result is None


def test_current_loop_variable_with_empty_sequence() -> None:
    """Test _current_loop_variable with empty sequence."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=[[]],
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.END,
        index=0,
    )

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)

    assert result is None


def test_current_loop_variable_with_nested_sequences() -> None:
    """Test _current_loop_variable with nested sequences."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=[[{"key": "value1"}, {"key": "value2"}], [{"key": "value3"}]],
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.END,
        index=0,
    )

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)

    assert result == {"key": "value1"}


def test_current_loop_variable_with_mixed_types() -> None:
    """Test _current_loop_variable with mixed types in sequence."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=[["string", 42, {"dict": "value"}, [1, 2, 3]]],
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.END,
        index=0,
    )

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)

    assert result == "string"


def test_current_loop_variable_with_string_over() -> None:
    """Test _current_loop_variable when over is a string."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over="single_string",
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.START,
        index=0,
    )

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)

    assert result == "single_string"


def test_current_loop_variable_with_string_over_different_index() -> None:
    """Test _current_loop_variable when over is a string with different index."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over="single_string",
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.START,
        index=1,
    )

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)

    # When over is a string, it's treated as a single-element list, so index 1 should return None
    assert result is None


def test_current_loop_variable_with_reference_resolving_to_string() -> None:
    """Test _current_loop_variable when _resolve_references returns a string."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("previous_step"),
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.START,
        index=0,
    )

    mock_run_data = Mock()

    # Mock the _resolve_references method to return a string
    with patch.object(step, "_resolve_references", return_value="resolved_string"):
        result = step._current_loop_variable(mock_run_data)

    assert result == "resolved_string"


def test_current_loop_variable_with_reference_resolving_to_string_different_index() -> None:
    """Test _current_loop_variable when _resolve_references returns a string.

    Tests with different index value.
    """
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("previous_step"),
        loop_type=LoopType.FOR_EACH,
        loop_step_type=LoopStepType.START,
        index=1,
    )

    mock_run_data = Mock()

    # Mock the _resolve_references method to return a string
    with patch.object(step, "_resolve_references", return_value="resolved_string"):
        result = step._current_loop_variable(mock_run_data)

    # When resolved value is a string, it's treated as a single-element list,
    # so index 1 should return None
    assert result is None
