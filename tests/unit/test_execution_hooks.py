"""Tests for the execution hooks functions in portia/execution_hooks.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from portia.clarification import ClarificationCategory, UserVerificationClarification
from portia.errors import ToolHardError
from portia.execution_agents.output import LocalDataValue
from portia.execution_hooks import (
    clarify_on_all_tool_calls,
    clarify_on_tool_calls,
    log_step_outputs,
)
from portia.plan import Step
from tests.utils import AdditionTool, ClarificationTool, get_test_plan_run

if TYPE_CHECKING:
    from portia.tool import Tool


def test_clarify_before_tool_call_first_call() -> None:
    """Test the cli_user_verify_before_tool_call hook on first call."""
    tool = AdditionTool()
    args = {"a": 1, "b": 2}
    plan, plan_run = get_test_plan_run()
    step = plan.steps[0]

    result = clarify_on_all_tool_calls(tool, args, plan_run, step)
    assert isinstance(result, UserVerificationClarification)
    assert result.category == ClarificationCategory.USER_VERIFICATION


def test_clarify_before_tool_call_with_previous_yes_response() -> None:
    """Test the cli_user_verify_before_tool_call hook with a previous 'yes' response."""
    tool = AdditionTool()
    args = {"a": 1, "b": 2}
    plan, plan_run = get_test_plan_run()
    step = plan.steps[0]

    # Create a previous clarification with 'yes' response
    prev_clarification = UserVerificationClarification(
        plan_run_id=plan_run.id,
        user_guidance=f"Are you happy to proceed with the call to {tool.name} with args {args}? "
        "Enter 'y' or 'yes' to proceed",
        resolved=True,
        response=True,
        step=0,
        source="Test execution hooks",
    )
    plan_run.outputs.clarifications = [prev_clarification]

    result = clarify_on_all_tool_calls(tool, args, plan_run, step)
    assert result is None


def test_clarify_before_tool_call_with_previous_no_response() -> None:
    """Test the cli_user_verify_before_tool_call hook with a previous 'no' response."""
    tool = AdditionTool()
    args = {"a": 1, "b": 2}
    plan, plan_run = get_test_plan_run()
    step = plan.steps[0]

    # Create a previous clarification with 'no' response
    prev_clarification = UserVerificationClarification(
        plan_run_id=plan_run.id,
        user_guidance=f"Are you happy to proceed with the call to {tool.name} with args {args}? "
        "Enter 'y' or 'yes' to proceed",
        resolved=True,
        response=False,
        step=0,
        source="Test execution hooks",
    )
    plan_run.outputs.clarifications = [prev_clarification]

    with pytest.raises(ToolHardError):
        clarify_on_all_tool_calls(tool, args, plan_run, step)


def test_clarify_before_tool_call_with_unresolved_clarification() -> None:
    """Test the cli_user_verify_before_tool_call hook with a previous unresolved clarification."""
    tool = AdditionTool()
    args = {"a": 1, "b": 2}
    plan, plan_run = get_test_plan_run()
    step = plan.steps[0]

    prev_clarification = UserVerificationClarification(
        plan_run_id=plan_run.id,
        user_guidance=f"Are you happy to proceed with the call to {tool.name} with args {args}? "
        "Enter 'y' or 'yes' to proceed",
        resolved=False,
        source="Test execution hooks",
    )
    plan_run.outputs.clarifications = [prev_clarification]

    # Call the hook and check we receive another clarification
    result = clarify_on_all_tool_calls(tool, args, plan_run, step)
    assert isinstance(result, UserVerificationClarification)


def test_log_step_outputs() -> None:
    """Test the log_step_outputs function."""
    plan, plan_run = get_test_plan_run()
    step = Step(task="Test task", tool_id="test_tool", output="$output")
    output = LocalDataValue(value="Test output", summary="Test summary")

    # Check it can be run without raising an error
    log_step_outputs(plan, plan_run, step, output)


@pytest.mark.parametrize(
    ("tool_id", "tool_to_test", "should_raise"),
    [
        # Single string tool ID cases
        ("test_tool", "test_tool", True),
        ("test_tool", "different_tool", False),
        # List of string tool IDs cases
        (["test_tool", "other_tool"], "test_tool", True),
        (["test_tool", "other_tool"], "other_tool", True),
        (["test_tool", "other_tool"], "different_tool", False),
        # Single Tool object cases
        (AdditionTool(), "add_tool", True),
        (AdditionTool(), "different_tool", False),
        # List of Tool objects cases
        ([AdditionTool(), ClarificationTool()], "add_tool", True),
        ([AdditionTool(), ClarificationTool()], "clarification_tool", True),
        ([AdditionTool(), ClarificationTool()], "different_tool", False),
    ],
)
def test_clarify_on_tool_calls_first_call(
    tool_id: str | list[str] | Tool | list[Tool], tool_to_test: str, *, should_raise: bool
) -> None:
    """Test clarify_on_tool_calls on first call with different tool IDs and Tool objects."""
    tool = AdditionTool()
    tool.id = tool_to_test
    args = {"a": 1, "b": 2}
    plan, plan_run = get_test_plan_run()
    step = plan.steps[0]

    hook = clarify_on_tool_calls(tool_id)
    result = hook(tool, args, plan_run, step)

    if should_raise:
        assert isinstance(result, UserVerificationClarification)
        assert result.category == ClarificationCategory.USER_VERIFICATION
        assert result.user_guidance is not None
    else:
        assert result is None
