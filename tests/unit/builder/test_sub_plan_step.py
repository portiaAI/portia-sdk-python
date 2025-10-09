"""Test the SubPlanStep class."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from portia.builder.reference import Input, StepOutput
from portia.builder.sub_plan_step import SubPlanStep
from portia.execution_agents.output import LocalDataValue
from portia.plan import PlanInput
from portia.plan import Step as PlanStep
from portia.run_context import StepOutputValue


def test_sub_plan_step_initialization() -> None:
    """Test SubPlanStep initialization."""
    mock_plan = Mock()
    input_values = {"param1": "value1", "param2": StepOutput(0)}

    step = SubPlanStep.model_construct(
        step_name="sub_plan_step",
        plan=mock_plan,
        input_values=input_values,
    )

    assert step.step_name == "sub_plan_step"
    assert step.plan == mock_plan


def test_sub_plan_step_initialization_defaults() -> None:
    """Test SubPlanStep initialization with default values."""
    mock_plan = Mock()

    step = SubPlanStep.model_construct(
        step_name="sub_plan_step",
        plan=mock_plan,
    )

    assert step.step_name == "sub_plan_step"
    assert step.plan == mock_plan


def test_sub_plan_step_str_with_label() -> None:
    """Test SubPlanStep str method with plan label."""
    mock_plan = Mock()
    mock_plan.label = "Test Plan Label"

    step = SubPlanStep.model_construct(
        step_name="sub_plan_step",
        plan=mock_plan,
    )

    assert str(step) == "SubPlanStep(plan='Test Plan Label')"


def test_sub_plan_step_str_without_label() -> None:
    """Test SubPlanStep str method without plan label."""
    mock_plan = Mock()
    mock_plan.id = "test_plan_id"
    mock_plan.label = None

    step = SubPlanStep.model_construct(
        step_name="sub_plan_step",
        plan=mock_plan,
    )

    assert str(step) == "SubPlanStep(plan='test_plan_id')"


@pytest.mark.asyncio
async def test_sub_plan_step_run_no_inputs() -> None:
    """Test SubPlanStep run with no input values."""
    mock_plan = Mock()
    mock_plan.plan_inputs = []

    step = SubPlanStep.model_construct(step_name="sub_plan_step", plan=mock_plan)

    mock_run_data = Mock()
    mock_run_data.end_user = Mock()
    mock_run_data.plan_run.plan_run_inputs = {}

    mock_plan_run = Mock()
    mock_final_output = Mock()
    mock_final_output.full_value.return_value = "Sub-plan completed successfully"
    mock_plan_run.outputs.final_output = mock_final_output

    with patch("portia.builder.sub_plan_step.Portia") as mock_portia_class:
        mock_portia = Mock()
        mock_portia.arun_plan = AsyncMock(return_value=mock_plan_run)
        mock_portia_class.return_value = mock_portia

        result = await step.run(run_data=mock_run_data)

        assert result == "Sub-plan completed successfully"
        mock_portia_class.assert_called_once()
        mock_portia.arun_plan.assert_called_once_with(
            mock_plan, mock_run_data.end_user, plan_run_inputs=[]
        )


@pytest.mark.asyncio
async def test_sub_plan_step_run_with_input_values() -> None:
    """Test SubPlanStep run with input values from step parameters."""
    mock_plan = Mock()
    mock_plan.plan_inputs = [
        PlanInput(name="username", description="User name", value="Alice"),
        PlanInput(name="data", description="Previous step data", value=StepOutput(0)),
    ]

    step = SubPlanStep.model_construct(step_name="sub_plan_step", plan=mock_plan)
    mock_plan.steps = [step]

    mock_run_data = Mock()
    mock_run_data.plan = mock_plan
    mock_run_data.plan_run.plan_run_inputs = {}
    mock_run_data.step_output_values = [
        StepOutputValue(
            value="previous step result",
            description="Step 0",
            step_name="previous_step",
            step_num=0,
        )
    ]

    mock_plan_run = Mock()
    mock_final_output = Mock()
    mock_final_output.full_value.return_value = "Sub-plan with inputs completed"
    mock_plan_run.outputs.final_output = mock_final_output

    with (
        patch("portia.builder.sub_plan_step.Portia") as mock_portia_class,
    ):
        mock_portia = Mock()
        mock_portia.arun_plan = AsyncMock(return_value=mock_plan_run)
        mock_portia_class.return_value = mock_portia

        result = await step.run(run_data=mock_run_data)

        assert result == "Sub-plan with inputs completed"

        # Verify that the plan inputs were correctly constructed
        call_args = mock_portia.arun_plan.call_args
        plan_run_inputs = call_args[1]["plan_run_inputs"]
        assert len(plan_run_inputs) == 2

        # Check username input
        username_input = next(inp for inp in plan_run_inputs if inp.name == "username")
        assert username_input.value == "Alice"

        # Check data input
        data_input = next(inp for inp in plan_run_inputs if inp.name == "data")
        assert data_input.value == "previous step result"


@pytest.mark.asyncio
async def test_sub_plan_step_run_input_priority() -> None:
    """Test SubPlanStep run input priority: step input_values > parent inputs > defaults."""
    mock_plan = Mock()
    mock_plan.plan_inputs = [
        PlanInput(name="param1", description="Parameter 1", value="default1"),
        PlanInput(name="param2", description="Parameter 2", value="default2"),
    ]

    step = SubPlanStep.model_construct(
        step_name="sub_plan_step",
        plan=mock_plan,
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.plan_run_inputs = {"param1": "parent_value1"}

    mock_plan_run = Mock()
    mock_final_output = Mock()
    mock_final_output.full_value.return_value = "Sub-plan with priority completed"
    mock_plan_run.outputs.final_output = mock_final_output

    with patch("portia.builder.sub_plan_step.Portia") as mock_portia_class:
        mock_portia = Mock()
        mock_portia.arun_plan = AsyncMock(return_value=mock_plan_run)
        mock_portia_class.return_value = mock_portia

        result = await step.run(run_data=mock_run_data)

        assert result == "Sub-plan with priority completed"

        # Verify input priority
        call_args = mock_portia.arun_plan.call_args
        plan_run_inputs = call_args[1]["plan_run_inputs"]
        assert len(plan_run_inputs) == 2

        # param1: plan run value should win
        param1 = next(inp for inp in plan_run_inputs if inp.name == "param1")
        assert param1.value == "parent_value1"

        # param2: default value should be used
        param2 = next(inp for inp in plan_run_inputs if inp.name == "param2")
        assert param2.value == "default2"


@pytest.mark.asyncio
async def test_sub_plan_step_run_no_final_output() -> None:
    """Test SubPlanStep run when sub-plan has no final output."""
    mock_plan = Mock()
    mock_plan.plan_inputs = []

    step = SubPlanStep.model_construct(
        step_name="sub_plan_step",
        plan=mock_plan,
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.plan_run_inputs = {}

    mock_plan_run = Mock()
    mock_plan_run.outputs.final_output = None  # No final output

    with patch("portia.builder.sub_plan_step.Portia") as mock_portia_class:
        mock_portia = Mock()
        mock_portia.arun_plan = AsyncMock(return_value=mock_plan_run)
        mock_portia_class.return_value = mock_portia

        result = await step.run(run_data=mock_run_data)

        assert result is None


@pytest.mark.asyncio
async def test_sub_plan_step_run_with_string_template_input_values() -> None:
    """Test SubPlanStep run with string template input values containing references."""
    mock_plan = Mock()
    mock_plan.plan_inputs = [
        PlanInput(
            name="message",
            description="Templated message",
            value=f"User {Input('username')} says: {StepOutput(0)}",
        )
    ]

    step = SubPlanStep.model_construct(
        step_name="sub_plan_step",
        plan=mock_plan,
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.plan_run_inputs = {"username": LocalDataValue(value="Alice")}
    mock_run_data.step_output_values = [
        StepOutputValue(
            value="Hello world",
            description="Step 0",
            step_name="greeting",
            step_num=0,
        )
    ]
    mock_run_data.plan = Mock()
    mock_run_data.plan.plan_inputs = [PlanInput(name="username")]
    mock_run_data.plan.steps = [step]

    mock_plan_run = Mock()
    mock_final_output = Mock()
    mock_final_output.full_value.return_value = "Sub-plan with templates completed"
    mock_plan_run.outputs.final_output = mock_final_output

    with patch("portia.builder.sub_plan_step.Portia") as mock_portia_class:
        mock_portia = Mock()
        mock_portia.arun_plan = AsyncMock(return_value=mock_plan_run)
        mock_portia_class.return_value = mock_portia

        result = await step.run(run_data=mock_run_data)

        assert result == "Sub-plan with templates completed"

        # Verify that templates were resolved
        call_args = mock_portia.arun_plan.call_args
        plan_run_inputs = call_args[1]["plan_run_inputs"]
        assert len(plan_run_inputs) == 1

        message_input = plan_run_inputs[0]
        assert message_input.name == "message"
        assert message_input.value == "User Alice says: Hello world"


def test_sub_plan_step_to_step_data() -> None:
    """Test SubPlanStep to_step_data method."""
    mock_sub_plan = Mock()
    mock_sub_plan.id = "sub_plan_id"
    mock_sub_plan.label = "Sub Plan"
    mock_sub_plan.steps = []
    mock_sub_plan.plan_inputs = [PlanInput(name="user_input")]

    mock_step1 = Mock()
    mock_legacy_step1 = Mock()
    mock_legacy_step1.tool_id = "tool1"
    mock_step1.to_step_data.return_value = mock_legacy_step1

    mock_step2 = Mock()
    mock_legacy_step2 = Mock()
    mock_legacy_step2.tool_id = None
    mock_step2.to_step_data.return_value = mock_legacy_step2

    mock_step3 = Mock()
    mock_legacy_step3 = Mock()
    mock_legacy_step3.tool_id = "tool3"
    mock_step3.to_step_data.return_value = mock_legacy_step3

    mock_sub_plan.steps = [mock_step1, mock_step2, mock_step3]

    step = SubPlanStep.model_construct(
        step_name="sub_plan_step",
        plan=mock_sub_plan,
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$sub_plan_step_output"

    legacy_step = step.to_step_data(mock_plan)

    assert isinstance(legacy_step, PlanStep)
    assert legacy_step.task == "Run sub-plan: Sub Plan"
    assert legacy_step.tool_id == "tool1,tool3"  # Only non-None tool_ids
    assert legacy_step.output == "$sub_plan_step_output"
    assert legacy_step.condition is None

    # Verify inputs conversion
    assert len(legacy_step.inputs) == 1
    assert legacy_step.inputs[0].name == "user_input"


def test_sub_plan_step_to_step_data_no_tools() -> None:
    """Test SubPlanStep to_step_data method when sub-plan has no tools."""
    mock_sub_plan = Mock()
    mock_sub_plan.id = "sub_plan_id"
    mock_sub_plan.label = "Sub Plan No Tools"
    mock_sub_plan.steps = []
    mock_sub_plan.plan_inputs = []

    # Create mock steps with no tool_ids
    mock_step1 = Mock()
    mock_legacy_step1 = Mock()
    mock_legacy_step1.tool_id = None
    mock_step1.to_step_data.return_value = mock_legacy_step1

    mock_step2 = Mock()
    mock_legacy_step2 = Mock()
    mock_legacy_step2.tool_id = None
    mock_step2.to_step_data.return_value = mock_legacy_step2

    mock_sub_plan.steps = [mock_step1, mock_step2]

    step = SubPlanStep.model_construct(
        step_name="sub_plan_step",
        plan=mock_sub_plan,
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$sub_plan_step_output"

    legacy_step = step.to_step_data(mock_plan)

    assert isinstance(legacy_step, PlanStep)
    assert legacy_step.task == "Run sub-plan: Sub Plan No Tools"
    assert legacy_step.tool_id == ""  # Empty string when no tools
    assert legacy_step.output == "$sub_plan_step_output"
    assert legacy_step.inputs == []
