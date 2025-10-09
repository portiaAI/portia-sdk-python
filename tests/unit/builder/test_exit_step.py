"""Test the step_v2 module."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from portia.builder.exit import ExitStepResult
from portia.builder.exit_step import ExitStep
from portia.plan import Step as PlanStep
from portia.run_context import StepOutputValue


def test_exit_step_initialization_defaults() -> None:
    """Test ExitStep initialization with default values."""
    step = ExitStep(step_name="exit_step")

    assert step.step_name == "exit_step"
    assert step.message == ""
    assert step.error is False
    assert step.conditional_block is None


def test_exit_step_initialization_with_custom_values() -> None:
    """Test ExitStep initialization with custom values."""
    step = ExitStep(step_name="error_exit", message="Something went wrong", error=True)

    assert step.step_name == "error_exit"
    assert step.message == "Something went wrong"
    assert step.error is True


def test_exit_step_str_representation_no_error() -> None:
    """Test ExitStep string representation without error."""
    step = ExitStep(step_name="exit_step", message="Normal exit")

    assert str(step) == "ExitStep - Normal exit"


def test_exit_step_str_representation_with_error() -> None:
    """Test ExitStep string representation with error."""
    step = ExitStep(step_name="exit_step", message="Error occurred", error=True)

    assert str(step) == "ExitStep (ERROR) - Error occurred"


def test_exit_step_str_representation_no_message() -> None:
    """Test ExitStep string representation without message."""
    step = ExitStep(step_name="exit_step")

    assert str(step) == "ExitStep"


@pytest.mark.asyncio
async def test_exit_step_run_basic() -> None:
    """Test ExitStep run method with basic message."""
    step = ExitStep(step_name="exit_step", message="Test exit")
    mock_run_data = Mock()

    result = await step.run(mock_run_data)

    assert isinstance(result, ExitStepResult)
    assert result.message == "Test exit"
    assert result.error is False


@pytest.mark.asyncio
async def test_exit_step_run_with_error() -> None:
    """Test ExitStep run method with error flag."""
    step = ExitStep(step_name="exit_step", message="Error exit", error=True)
    mock_run_data = Mock()

    result = await step.run(mock_run_data)

    assert isinstance(result, ExitStepResult)
    assert result.message == "Error exit"
    assert result.error is True


@pytest.mark.asyncio
async def test_exit_step_run_with_reference_resolution() -> None:
    """Test ExitStep run method with message reference resolution."""
    step = ExitStep(step_name="exit_step", message="Processing {{ StepOutput(0) }}")

    # Create a proper RunContext with step_output_values populated
    mock_storage = Mock()
    mock_portia = Mock()
    mock_portia.storage = mock_storage

    mock_step_output_value = StepOutputValue(
        step_num=0, step_name="step_0", value="completed successfully", description="Step 0 output"
    )

    mock_run_data = Mock()
    mock_run_data.step_output_values = [mock_step_output_value]
    mock_run_data.portia = mock_portia
    mock_run_data.plan.steps = [step]

    result = await step.run(mock_run_data)

    assert isinstance(result, ExitStepResult)
    assert result.message == "Processing completed successfully"
    assert result.error is False


@pytest.mark.asyncio
async def test_exit_step_run_with_non_string_resolved_message() -> None:
    """Test ExitStep run method when resolved message is not a string."""
    step = ExitStep(step_name="exit_step", message="Processing {{ StepOutput(0) }}")

    # Create a proper RunContext with step_output_values populated
    mock_storage = Mock()
    mock_portia = Mock()
    mock_portia.storage = mock_storage

    # Create a StepOutputValue containing a non-string value
    mock_step_output_value = StepOutputValue(
        step_num=0, step_name="step_0", value=42, description="Step 0 output"
    )

    mock_run_data = Mock()
    mock_run_data.step_output_values = [mock_step_output_value]
    mock_run_data.portia = mock_portia
    mock_run_data.plan.steps = [step]

    result = await step.run(mock_run_data)

    assert isinstance(result, ExitStepResult)
    assert result.message == "Processing 42"
    assert result.error is False


@pytest.mark.asyncio
async def test_exit_step_run_with_none_resolved_message() -> None:
    """Test ExitStep run method when resolved message is None."""
    step = ExitStep(step_name="exit_step", message="Processing {{ StepOutput(0) }}")

    # Create a proper RunContext with step_output_values populated
    mock_storage = Mock()
    mock_portia = Mock()
    mock_portia.storage = mock_storage

    # Create a StepOutputValue containing None value
    mock_step_output_value = StepOutputValue(
        step_num=0, step_name="step_0", value=None, description="Step 0 output"
    )

    mock_run_data = Mock()
    mock_run_data.step_output_values = [mock_step_output_value]
    mock_run_data.portia = mock_portia
    mock_run_data.plan.steps = [step]

    result = await step.run(mock_run_data)

    assert isinstance(result, ExitStepResult)
    assert result.message == "Processing None"
    assert result.error is False


def test_exit_step_to_step_data_normal_exit() -> None:
    """Test ExitStep to_step_data method for normal exit."""
    step = ExitStep(step_name="exit_step", message="Normal completion")
    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$exit_step_output"

    legacy_step = step.to_step_data(mock_plan)

    assert isinstance(legacy_step, PlanStep)
    assert legacy_step.task == "Exit plan: Normal completion"
    assert legacy_step.tool_id is None
    assert legacy_step.output == "$exit_step_output"
    assert legacy_step.structured_output_schema is None
    assert len(legacy_step.inputs) == 0


def test_exit_step_to_step_data_error_exit() -> None:
    """Test ExitStep to_step_data method for error exit."""
    step = ExitStep(step_name="exit_step", message="Error occurred", error=True)
    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$exit_step_output"

    legacy_step = step.to_step_data(mock_plan)

    assert isinstance(legacy_step, PlanStep)
    assert legacy_step.task == "Exit plan with error: Error occurred"
    assert legacy_step.tool_id is None
    assert legacy_step.output == "$exit_step_output"
    assert legacy_step.structured_output_schema is None
    assert len(legacy_step.inputs) == 0


def test_exit_step_to_step_data_no_message() -> None:
    """Test ExitStep to_step_data method without message."""
    step = ExitStep(step_name="exit_step")
    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$exit_step_output"

    legacy_step = step.to_step_data(mock_plan)

    assert isinstance(legacy_step, PlanStep)
    assert legacy_step.task == "Exit plan: "
    assert legacy_step.tool_id is None
    assert legacy_step.output == "$exit_step_output"
    assert legacy_step.structured_output_schema is None
    assert len(legacy_step.inputs) == 0
