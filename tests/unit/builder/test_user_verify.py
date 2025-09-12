"""Test the user verify step."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from portia.builder.reference import Input, StepOutput
from portia.builder.user_verify import UserVerifyStep
from portia.clarification import (
    UserVerificationClarification,
)
from portia.errors import PlanRunExitError
from portia.execution_agents.output import LocalDataValue
from portia.plan import PlanInput
from portia.plan import Step as PlanStep
from portia.prefixed_uuid import PlanRunUUID
from portia.run_context import StepOutputValue


def test_user_verify_step_str() -> None:
    """Test UserVerifyStep str method."""
    step = UserVerifyStep(message="Please confirm this action", step_name="verify")
    assert str(step) == "UserVerifyStep(message='Please confirm this action')"


def test_user_verify_step_to_legacy_step() -> None:
    """Test UserVerifyStep to_legacy_step method."""
    step = UserVerifyStep(message="Confirm deletion", step_name="confirm_delete")

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$confirm_delete_output"

    legacy_step = step.to_legacy_step(mock_plan)

    assert isinstance(legacy_step, PlanStep)
    assert legacy_step.task == "User verification: Confirm deletion"
    assert legacy_step.inputs == []
    assert legacy_step.tool_id is None
    assert legacy_step.output == "$confirm_delete_output"
    assert legacy_step.structured_output_schema is None


@pytest.mark.asyncio
async def test_user_verify_step_requests_clarification() -> None:
    """Test that UserVerifyStep returns a clarification on first run."""
    message = f"Proceed with {StepOutput(0)} for {Input('username')}?"
    step = UserVerifyStep(message=message, step_name="verify")

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.id = PlanRunUUID()
    mock_run_data.plan_run.get_clarification_for_step.return_value = None
    mock_run_data.plan = Mock()
    mock_run_data.plan.plan_inputs = [PlanInput(name="username")]
    mock_run_data.plan_run.plan_run_inputs = {"username": LocalDataValue(value="Alice")}
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_0", value="result", description="")
    ]

    result = await step.run(run_data=mock_run_data)

    assert isinstance(result, UserVerificationClarification)
    assert result.user_guidance == "Proceed with result for Alice?"


@pytest.mark.asyncio
async def test_user_verify_step_user_confirms() -> None:
    """Test that the step succeeds when user verifies."""
    step = UserVerifyStep(message="Confirm?", step_name="verify")

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    clarification = UserVerificationClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Confirm?",
        response=True,
        resolved=True,
    )
    mock_run_data.plan_run.get_clarification_for_step.return_value = clarification

    result = await step.run(run_data=mock_run_data)

    assert result is True


@pytest.mark.asyncio
async def test_user_verify_step_user_rejects() -> None:
    """Test that the step raises error when user rejects."""
    step = UserVerifyStep(message="Confirm?", step_name="verify")

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    clarification = UserVerificationClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Confirm?",
        response=False,
        resolved=True,
    )
    mock_run_data.plan_run.get_clarification_for_step.return_value = clarification

    with pytest.raises(PlanRunExitError):
        await step.run(run_data=mock_run_data)


@pytest.mark.asyncio
async def test_user_verify_step_with_string_template_message() -> None:
    """Test UserVerifyStep message resolution with reference templates."""
    step = UserVerifyStep(
        message=f"Confirm action on {StepOutput(0)} for user {Input('username')}?",
        step_name="verify_action",
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.id = PlanRunUUID()
    mock_run_data.plan_run.get_clarification_for_step.return_value = None
    mock_run_data.plan = Mock()
    mock_run_data.plan.plan_inputs = [PlanInput(name="username")]
    mock_run_data.plan_run.plan_run_inputs = {"username": LocalDataValue(value="Bob")}
    mock_run_data.step_output_values = [
        StepOutputValue(
            value="test_file.txt",
            description="step0",
            step_name="verify_action",
            step_num=0,
        )
    ]

    result = await step.run(run_data=mock_run_data)

    assert isinstance(result, UserVerificationClarification)
    assert result.user_guidance == "Confirm action on test_file.txt for user Bob?"
