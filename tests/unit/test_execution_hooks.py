"""Tests for the execution hooks functions in portia/execution_hooks.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from portia.builder.invoke_tool_step import InvokeToolStep
from portia.builder.plan_v2 import PlanV2
from portia.clarification import Clarification, ClarificationCategory, UserVerificationClarification
from portia.end_user import EndUser
from portia.errors import ToolHardError
from portia.execution_agents.output import LocalDataValue
from portia.execution_hooks import (
    clarify_on_all_tool_calls,
    clarify_on_tool_calls,
    log_step_outputs,
)
from portia.plan_run import PlanRunV2
from tests.utils import AdditionTool, ClarificationTool, get_test_config

if TYPE_CHECKING:
    from portia.tool import Tool


def get_test_plan_run_v2() -> tuple[PlanV2, PlanRunV2]:
    """Generate a simple test PlanRunV2."""
    step1 = InvokeToolStep(
        step_name="step_0",
        tool="add_tool",
        args={"a": 1, "b": 2},
    )
    plan_v2 = PlanV2(
        steps=[step1],
        label="Test plan",
    )
    end_user = EndUser(external_id="test_user")
    config = get_test_config()
    plan_run_v2 = PlanRunV2(
        plan=plan_v2,
        end_user=end_user,
        config=config,
    )
    return plan_v2, plan_run_v2


def test_clarify_before_tool_call_first_call() -> None:
    """Test the cli_user_verify_before_tool_call hook on first call."""
    tool = AdditionTool()
    args = {"a": 1, "b": 2}
    plan_v2, plan_run_v2 = get_test_plan_run_v2()
    step_v2 = plan_v2.steps[0]

    result = clarify_on_all_tool_calls(tool, args, plan_run_v2, step_v2)
    assert isinstance(result, UserVerificationClarification)
    assert result.category == ClarificationCategory.USER_VERIFICATION


def test_clarify_before_tool_call_with_previous_yes_response() -> None:
    """Test the cli_user_verify_before_tool_call hook with a previous 'yes' response."""
    tool = AdditionTool()
    args = {"a": 1, "b": 2}
    plan_v2, plan_run_v2 = get_test_plan_run_v2()
    step_v2 = plan_v2.steps[0]

    # Create a previous clarification with 'yes' response
    # Note: PlanRunV2.get_clarification_for_step currently returns None (TODO in implementation)
    # This test will need to be updated when clarification handling is implemented in V2
    prev_clarification = UserVerificationClarification(
        plan_run_id=plan_run_v2.id,
        user_guidance=f"Are you happy to proceed with the call to {tool.name} with args {args}? "
        "Enter 'y' or 'yes' to proceed",
        resolved=True,
        response=True,
        step=0,
        source="Test execution hooks",
    )
    # TODO: Add clarifications to PlanRunV2 when clarification handling is implemented
    # For now, this test will always return a clarification since get_clarification_for_step returns None

    result = clarify_on_all_tool_calls(tool, args, plan_run_v2, step_v2)
    # Once clarification handling is implemented, this should be None
    # For now, it will always return a clarification
    assert isinstance(result, UserVerificationClarification)


def test_clarify_before_tool_call_with_previous_negative_response() -> None:
    """Test the cli_user_verify_before_tool_call hook with a previous 'no' response."""
    tool = AdditionTool()
    args = {"a": 1, "b": 2}
    plan_v2, plan_run_v2 = get_test_plan_run_v2()
    step_v2 = plan_v2.steps[0]

    # Create a previous clarification with 'no' response
    # Note: This test is currently non-functional until clarification handling is implemented in V2
    prev_clarification = UserVerificationClarification(
        plan_run_id=plan_run_v2.id,
        user_guidance=f"Are you happy to proceed with the call to {tool.name} with args {args}? "
        "Enter 'y' or 'yes' to proceed",
        resolved=True,
        response=False,
        step=0,
        source="Test execution hooks",
    )
    # TODO: Add clarifications to PlanRunV2 when clarification handling is implemented

    # For now, since get_clarification_for_step returns None, it won't raise ToolHardError
    result = clarify_on_all_tool_calls(tool, args, plan_run_v2, step_v2)
    assert isinstance(result, UserVerificationClarification)


def test_clarify_before_tool_call_with_previous_negative_response_bare_clarification() -> None:
    """Test the cli_user_verify_before_tool_call hook with a previous 'no' response."""
    tool = AdditionTool()
    args = {"a": 1, "b": 2}
    plan_v2, plan_run_v2 = get_test_plan_run_v2()
    step_v2 = plan_v2.steps[0]

    # Create a previous clarification with 'no' response
    # This is a bare clarification, not a UserVerificationClarification, which reflects the
    # real runtime behaviour, where PlanRuns are serialised and deserialised
    # Note: This test is currently non-functional until clarification handling is implemented in V2
    prev_clarification = Clarification(
        plan_run_id=plan_run_v2.id,
        category=ClarificationCategory.USER_VERIFICATION,
        user_guidance=f"Are you happy to proceed with the call to {tool.name} with args {args}? "
        "Enter 'y' or 'yes' to proceed",
        resolved=True,
        response=False,
        step=0,
        source="Test execution hooks",
    )
    # TODO: Add clarifications to PlanRunV2 when clarification handling is implemented

    # For now, since get_clarification_for_step returns None, it won't raise ToolHardError
    result = clarify_on_all_tool_calls(tool, args, plan_run_v2, step_v2)
    assert isinstance(result, UserVerificationClarification)


def test_clarify_before_tool_call_with_unresolved_clarification() -> None:
    """Test the cli_user_verify_before_tool_call hook with a previous unresolved clarification."""
    tool = AdditionTool()
    args = {"a": 1, "b": 2}
    plan_v2, plan_run_v2 = get_test_plan_run_v2()
    step_v2 = plan_v2.steps[0]

    # Note: This test is currently non-functional until clarification handling is implemented in V2
    prev_clarification = UserVerificationClarification(
        plan_run_id=plan_run_v2.id,
        user_guidance=f"Are you happy to proceed with the call to {tool.name} with args {args}? "
        "Enter 'y' or 'yes' to proceed",
        resolved=False,
        source="Test execution hooks",
    )
    # TODO: Add clarifications to PlanRunV2 when clarification handling is implemented

    # Call the hook and check we receive another clarification
    result = clarify_on_all_tool_calls(tool, args, plan_run_v2, step_v2)
    assert isinstance(result, UserVerificationClarification)


def test_log_step_outputs() -> None:
    """Test the log_step_outputs function."""
    plan_v2, plan_run_v2 = get_test_plan_run_v2()
    step_v2 = plan_v2.steps[0]
    output = LocalDataValue(value="Test output", summary="Test summary")

    # Check it can be run without raising an error
    log_step_outputs(plan_run_v2, step_v2, output)


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
    plan_v2, plan_run_v2 = get_test_plan_run_v2()
    step_v2 = plan_v2.steps[0]

    hook = clarify_on_tool_calls(tool_id)
    result = hook(tool, args, plan_run_v2, step_v2)

    if should_raise:
        assert isinstance(result, UserVerificationClarification)
        assert result.category == ClarificationCategory.USER_VERIFICATION
        assert result.user_guidance is not None
    else:
        assert result is None
