"""Test the user input step."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from portia.builder.reference import Input, StepOutput
from portia.builder.user_input import UserInputStep
from portia.clarification import (
    InputClarification,
    MultipleChoiceClarification,
)
from portia.execution_agents.output import LocalDataValue
from portia.plan import PlanInput
from portia.plan import Step as PlanStep
from portia.prefixed_uuid import PlanRunUUID
from portia.run_context import StepOutputValue


def test_user_input_step_str_text_input() -> None:
    """Test UserInputStep str method for text input."""
    step = UserInputStep(
        message="Please provide input",
        step_name="input",
    )

    assert str(step) == "UserInputStep(type='text input', message='Please provide input')"


def test_user_input_step_str_multiple_choice() -> None:
    """Test UserInputStep str method for multiple choice."""
    step = UserInputStep(
        message="Choose an option",
        step_name="choice",
        options=["A", "B", "C"],
    )

    assert str(step) == "UserInputStep(type='multiple choice', message='Choose an option')"


def test_user_input_step_to_legacy_step_text_input() -> None:
    """Test UserInputStep to_legacy_step method for text input."""
    step = UserInputStep(
        message="Enter your name",
        step_name="name_input",
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$name_input_output"

    legacy_step = step.to_legacy_step(mock_plan)

    assert isinstance(legacy_step, PlanStep)
    assert legacy_step.task == "User input (Text input): Enter your name"
    assert legacy_step.inputs == []
    assert legacy_step.tool_id is None
    assert legacy_step.output == "$name_input_output"
    assert legacy_step.structured_output_schema is None


def test_user_input_step_to_legacy_step_multiple_choice() -> None:
    """Test UserInputStep to_legacy_step method for multiple choice."""
    step = UserInputStep(
        message="Choose an option",
        step_name="choice",
        options=["A", "B", "C"],
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$choice_output"

    legacy_step = step.to_legacy_step(mock_plan)

    assert isinstance(legacy_step, PlanStep)
    assert legacy_step.task == "User input (Multiple choice): Choose an option"
    assert legacy_step.inputs == []
    assert legacy_step.tool_id is None
    assert legacy_step.output == "$choice_output"
    assert legacy_step.structured_output_schema is None


@pytest.mark.asyncio
async def test_user_input_step_text_input_requests_clarification() -> None:
    """Test that UserInputStep returns INPUT clarification on first run for text input."""
    step = UserInputStep(
        message="Please enter your name",
        step_name="get_name",
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.id = PlanRunUUID()
    mock_run_data.plan_run.get_clarification_for_step.return_value = None
    mock_run_data.plan = Mock()
    mock_run_data.plan.step_output_name.return_value = "user_input"

    result = await step.run(run_data=mock_run_data)

    assert isinstance(result, InputClarification)
    assert result.user_guidance == "Please enter your name"
    assert result.argument_name == "user_input"


@pytest.mark.asyncio
async def test_user_input_step_multiple_choice_requests_clarification() -> None:
    """Test that UserInputStep returns MULTIPLE_CHOICE clarification when options provided."""
    step = UserInputStep(
        message="Choose your favorite color",
        step_name="choose_color",
        options=["red", "green", "blue"],
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.id = PlanRunUUID()
    mock_run_data.plan_run.get_clarification_for_step.return_value = None
    mock_run_data.plan = Mock()
    mock_run_data.plan.step_output_name.return_value = "user_input"

    result = await step.run(run_data=mock_run_data)

    assert isinstance(result, MultipleChoiceClarification)
    assert result.user_guidance == "Choose your favorite color"
    assert result.argument_name == "user_input"
    assert result.options == ["red", "green", "blue"]


@pytest.mark.asyncio
async def test_user_input_step_returns_response_when_resolved_text_input() -> None:
    """Test that UserInputStep returns response when text input clarification is resolved."""
    step = UserInputStep(
        message="Please enter your name",
        step_name="get_name",
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    clarification = InputClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Please enter your name",
        argument_name="user_input",
        response="Alice",
        resolved=True,
    )
    mock_run_data.plan_run.get_clarification_for_step.return_value = clarification

    result = await step.run(run_data=mock_run_data)

    assert result == "Alice"


@pytest.mark.asyncio
async def test_user_input_step_returns_response_when_resolved_multiple_choice() -> None:
    """Test UserInputStep returns response when multiple choice clarification resolved."""
    step = UserInputStep(
        message="Choose your favorite color",
        step_name="choose_color",
        options=["red", "green", "blue"],
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    clarification = MultipleChoiceClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Choose your favorite color",
        argument_name="user_input",
        options=["red", "green", "blue"],
        response="blue",
        resolved=True,
    )
    mock_run_data.plan_run.get_clarification_for_step.return_value = clarification

    result = await step.run(run_data=mock_run_data)

    assert result == "blue"


@pytest.mark.asyncio
async def test_user_input_step_message_with_templates() -> None:
    """Test that UserInputStep resolves references in the message and options."""
    step = UserInputStep(
        message=f"Provide feedback on {StepOutput(0)} by {Input('username')}",
        step_name="feedback",
        options=[
            "Good",
            f"Bad - {StepOutput(1)}",
            Input("Custom good phrase"),
            "Excellent",
        ],
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.id = PlanRunUUID()
    mock_run_data.plan_run.get_clarification_for_step.return_value = None
    mock_run_data.plan = Mock()
    mock_run_data.plan.step_output_name.return_value = "feedback"
    mock_run_data.plan.plan_inputs = [
        PlanInput(name="username"),
        PlanInput(name="Custom good phrase"),
    ]
    mock_run_data.plan_run.plan_run_inputs = {
        "username": LocalDataValue(value="Alice"),
        "Custom good phrase": LocalDataValue(value="Great job!"),
    }
    mock_run_data.step_output_values = [
        StepOutputValue(
            value="analysis result",
            description="s0",
            step_name="feedback",
            step_num=0,
        ),
        StepOutputValue(
            value="missing data",
            description="s1",
            step_name="feedback",
            step_num=1,
        ),
    ]
    mock_run_data.plan.steps = []
    mock_run_data.storage = Mock()

    result = await step.run(run_data=mock_run_data)

    assert isinstance(result, MultipleChoiceClarification)
    assert result.user_guidance == "Provide feedback on analysis result by Alice"
    assert result.options == ["Good", "Bad - missing data", "Great job!", "Excellent"]
