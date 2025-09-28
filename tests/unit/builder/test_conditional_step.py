"""Test the conditional step."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from portia.builder.conditional_step import ConditionalStep
from portia.builder.conditionals import (
    ConditionalBlock,
    ConditionalBlockClauseType,
    ConditionalStepResult,
)
from portia.builder.reference import Input, StepOutput
from portia.plan import Step as PlanStep


def test_conditional_step_good_initialization() -> None:
    """Test ConditionalStep initialization with valid parameters."""
    conditional_block = ConditionalBlock(
        clause_step_indexes=[0, 5, 10],
        parent_conditional_block=None,
    )

    def test_condition(x: int) -> bool:
        return x > 0

    step = ConditionalStep(
        step_name="test_conditional",
        conditional_block=conditional_block,
        condition=test_condition,
        args={"x": 5},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    assert step.step_name == "test_conditional"
    assert step.conditional_block == conditional_block
    assert step.condition == test_condition
    assert step.args == {"x": 5}
    assert step.clause_index_in_block == 0
    assert step.block_clause_type == ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK


def test_conditional_step_initialization_with_string_condition() -> None:
    """Test ConditionalStep initialization with string condition."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 3])

    step = ConditionalStep(
        step_name="string_conditional",
        conditional_block=conditional_block,
        condition="user_input > 10",
        args={"user_input": Input("number")},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    assert step.condition == "user_input > 10"


def test_conditional_step_validation_none_conditional_block() -> None:
    """Test that ConditionalStep raises ValueError when conditional_block is None."""

    def test_condition() -> bool:
        return True

    with pytest.raises(ValueError, match="Conditional block is required for ConditionSteps"):
        ConditionalStep(
            step_name="invalid_conditional",
            conditional_block=None,
            condition=test_condition,
            args={},
            clause_index_in_block=0,
            block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
        )


def test_conditional_step_block_property_good_case() -> None:
    """Test ConditionalStep block property returns the conditional block."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 2, 4])

    step = ConditionalStep(
        step_name="test_conditional",
        conditional_block=conditional_block,
        condition=lambda: True,
        args={},
        clause_index_in_block=1,
        block_clause_type=ConditionalBlockClauseType.ALTERNATE_CLAUSE,
    )

    assert step.block == conditional_block
    assert isinstance(step.block, ConditionalBlock)


def test_conditional_step_block_property_error_case() -> None:
    """Test ConditionalStep block property raises error for invalid conditional_block type."""
    # This test creates an invalid state that shouldn't normally occur due to validation
    step = ConditionalStep(
        step_name="test_conditional",
        conditional_block=ConditionalBlock(clause_step_indexes=[0]),
        condition=lambda: True,
        args={},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    # Manually set conditional_block to invalid type to test error case
    step.conditional_block = "invalid_type"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="Conditional block is not a ConditionalBlock"):
        _ = step.block


def test_conditional_step_str() -> None:
    """Test ConditionalStep str method."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 3])

    def test_function() -> bool:
        return True

    step = ConditionalStep(
        step_name="test_conditional",
        conditional_block=conditional_block,
        condition=test_function,
        args={"value": 42},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    result = str(step)
    assert "ConditionalStep" in result
    assert "test_function" in result or str(test_function) in result
    assert "NEW_CONDITIONAL_BLOCK" in result
    assert "{'value': 42}" in result


@pytest.mark.asyncio
async def test_conditional_step_run_with_function_condition_true() -> None:
    """Test ConditionalStep run with function condition that returns True."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 5, 10, 15])

    def test_condition(x: int) -> bool:
        return x > 5

    step = ConditionalStep(
        step_name="test_conditional",
        conditional_block=conditional_block,
        condition=test_condition,
        args={"x": 10},
        clause_index_in_block=1,  # Second clause (else_if)
        block_clause_type=ConditionalBlockClauseType.ALTERNATE_CLAUSE,
    )

    mock_run_data = Mock()

    result = await step.run(run_data=mock_run_data)

    assert isinstance(result, ConditionalStepResult)
    assert result.type == ConditionalBlockClauseType.ALTERNATE_CLAUSE
    assert result.conditional_result is True
    assert result.next_clause_step_index == 10  # Next clause
    assert result.end_condition_block_step_index == 15  # Last clause (endif)


@pytest.mark.asyncio
async def test_conditional_step_run_with_function_condition_false() -> None:
    """Test ConditionalStep run with function condition that returns False."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 3])

    def test_condition(x: int) -> bool:
        return x > 10

    step = ConditionalStep(
        step_name="test_conditional",
        conditional_block=conditional_block,
        condition=test_condition,
        args={"x": 5},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    mock_run_data = Mock()

    result = await step.run(run_data=mock_run_data)

    assert isinstance(result, ConditionalStepResult)
    assert result.conditional_result is False
    assert result.next_clause_step_index == 3
    assert result.end_condition_block_step_index == 3


@pytest.mark.asyncio
async def test_conditional_step_run_with_reference_args() -> None:
    """Test ConditionalStep run with reference arguments."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 4])
    reference_input = StepOutput(0)

    def test_condition(value: int) -> bool:
        return value > 0

    step = ConditionalStep(
        step_name="test_conditional",
        conditional_block=conditional_block,
        condition=test_condition,
        args={"value": reference_input},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    mock_run_data = Mock()
    mock_run_data.portia.storage = Mock()

    with patch.object(reference_input, "get_value") as mock_get_value:
        mock_get_value.return_value = 42

        result = await step.run(run_data=mock_run_data)

        assert isinstance(result, ConditionalStepResult)
        assert result.conditional_result is True
        mock_get_value.assert_called_once_with(mock_run_data)


@pytest.mark.asyncio
async def test_conditional_step_run_with_string_condition() -> None:
    """Test ConditionalStep run with string condition."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 2])

    step = ConditionalStep(
        step_name="string_conditional",
        conditional_block=conditional_block,
        condition="x > 5",
        args={"x": 10},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    mock_run_data = Mock()
    mock_agent = Mock()
    mock_agent.execute = AsyncMock(return_value=True)

    with patch("portia.builder.conditional_step.ConditionalEvaluationAgent") as mock_agent_class:
        mock_agent_class.return_value = mock_agent

        result = await step.run(run_data=mock_run_data)

        assert isinstance(result, ConditionalStepResult)
        assert result.conditional_result is True
        mock_agent_class.assert_called_once_with(mock_run_data.config)
        mock_agent.execute.assert_called_once_with(conditional="x > 5", arguments={"x": 10})


def test_conditional_step_to_legacy_step_with_function_condition() -> None:
    """Test ConditionalStep to_legacy_step with function condition."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 3])

    def my_test_function(x: int) -> bool:
        return x > 0

    step = ConditionalStep(
        step_name="function_conditional",
        conditional_block=conditional_block,
        condition=my_test_function,
        args={"x": Input("user_input")},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$function_conditional_output"

    with (
        patch.object(step.args["x"], "get_legacy_name") as mock_input_name,
        patch.object(step, "_get_legacy_condition") as mock_legacy_condition,
    ):
        mock_input_name.return_value = "user_input"
        mock_legacy_condition.return_value = "test condition"

        legacy_step = step.to_legacy_step(mock_plan)

        assert isinstance(legacy_step, PlanStep)
        assert legacy_step.task == "Conditional clause: If result of my_test_function is true"
        assert legacy_step.tool_id is None
        assert legacy_step.output == "$function_conditional_output"
        assert legacy_step.condition == "test condition"

        # Verify inputs conversion
        assert len(legacy_step.inputs) == 1
        assert legacy_step.inputs[0].name == "user_input"

        mock_legacy_condition.assert_called_once_with(mock_plan)


def test_conditional_step_to_legacy_step_with_string_condition() -> None:
    """Test ConditionalStep to_legacy_step with string condition."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 2])

    step = ConditionalStep(
        step_name="string_conditional",
        conditional_block=conditional_block,
        condition="user_age >= 18",
        args={"user_age": Input("age")},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$string_conditional_output"

    with (
        patch.object(step.args["user_age"], "get_legacy_name") as mock_input_name,
        patch.object(step, "_get_legacy_condition") as mock_legacy_condition,
    ):
        mock_input_name.return_value = "age"
        mock_legacy_condition.return_value = "legacy condition string"

        legacy_step = step.to_legacy_step(mock_plan)

        assert isinstance(legacy_step, PlanStep)
        assert legacy_step.task == "Conditional clause: user_age >= 18"
        assert legacy_step.tool_id is None
        assert legacy_step.output == "$string_conditional_output"
        assert legacy_step.condition == "legacy condition string"

        assert len(legacy_step.inputs) == 1
        assert legacy_step.inputs[0].name == "age"


def test_conditional_step_to_legacy_step_with_lambda_function() -> None:
    """Test ConditionalStep to_legacy_step with lambda function condition."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 1])

    lambda_condition = lambda x: x > 5  # noqa: E731

    step = ConditionalStep(
        step_name="lambda_conditional",
        conditional_block=conditional_block,
        condition=lambda_condition,
        args={},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$lambda_conditional_output"

    with patch.object(step, "_get_legacy_condition") as mock_legacy_condition:
        mock_legacy_condition.return_value = None

        legacy_step = step.to_legacy_step(mock_plan)

        assert "If result of" in legacy_step.task
        assert "is true" in legacy_step.task
