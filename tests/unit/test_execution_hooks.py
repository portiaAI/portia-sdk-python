"""Tests for the execution hooks functions in portia/execution_hooks.py."""

from __future__ import annotations

import pytest

from portia.clarification import ClarificationCategory, CustomClarification
from portia.errors import ToolHardError
from portia.execution_agents.output import LocalDataValue
from portia.execution_hooks import (
    clarify_on_all_tool_calls,
    clarify_on_tool_calls,
    log_step_outputs,
)
from portia.plan import Step
from tests.utils import AdditionTool, get_test_plan_run


def test_cli_user_verify_before_tool_call_first_call() -> None:
    """Test the cli_user_verify_before_tool_call hook on first call."""
    tool = AdditionTool()
    args = {"a": 1, "b": 2}
    plan, plan_run = get_test_plan_run()
    step = plan.steps[0]

    result = clarify_on_all_tool_calls(tool, args, plan_run, step)
    assert isinstance(result, CustomClarification)
    assert result.category == ClarificationCategory.CUSTOM


def test_cli_user_verify_before_tool_call_with_previous_yes_response() -> None:
    """Test the cli_user_verify_before_tool_call hook with a previous 'yes' response."""
    tool = AdditionTool()
    args = {"a": 1, "b": 2}
    plan, plan_run = get_test_plan_run()
    step = plan.steps[0]

    # Create a previous clarification with 'yes' response
    prev_clarification = CustomClarification(
        name="user_verify",
        plan_run_id=plan_run.id,
        user_guidance=f"Are you happy to proceed with the call to {tool.name}? "
        "Enter 'y' or 'yes' to proceed",
        data={"args": args},
        resolved=True,
        response="yes",
        step=0,
    )
    plan_run.outputs.clarifications = [prev_clarification]

    result = clarify_on_all_tool_calls(tool, args, plan_run, step)
    assert result is None


def test_cli_user_verify_before_tool_call_with_previous_no_response() -> None:
    """Test the cli_user_verify_before_tool_call hook with a previous 'no' response."""
    tool = AdditionTool()
    args = {"a": 1, "b": 2}
    plan, plan_run = get_test_plan_run()
    step = plan.steps[0]

    # Create a previous clarification with 'no' response
    prev_clarification = CustomClarification(
        name="user_verify",
        plan_run_id=plan_run.id,
        user_guidance=f"Are you happy to proceed with the call to {tool.name}? "
        "Enter 'y' or 'yes' to proceed",
        data={"args": args},
        resolved=True,
        response="no",
        step=0,
    )
    plan_run.outputs.clarifications = [prev_clarification]

    with pytest.raises(ToolHardError):
        clarify_on_all_tool_calls(tool, args, plan_run, step)


def test_cli_user_verify_before_tool_call_with_unresolved_clarification() -> None:
    """Test the cli_user_verify_before_tool_call hook with a previous unresolved clarification."""
    tool = AdditionTool()
    args = {"a": 1, "b": 2}
    plan, plan_run = get_test_plan_run()
    step = plan.steps[0]

    prev_clarification = CustomClarification(
        name="user_verify",
        plan_run_id=plan_run.id,
        user_guidance=f"Are you happy to proceed with the call to {tool.name}? "
        "Enter 'y' or 'yes' to proceed",
        data={"args": args},
        resolved=False,
        step=0,
    )
    plan_run.outputs.clarifications = [prev_clarification]

    # Call the hook and check we reveive another clarification
    result = clarify_on_all_tool_calls(tool, args, plan_run, step)
    assert isinstance(result, CustomClarification)


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
        ("test_tool", "test_tool", True),
        (["test_tool", "other_tool"], "test_tool", True),
        (["test_tool", "other_tool"], "other_tool", True),
        ("test_tool", "different_tool", False),
        (["test_tool", "other_tool"], "different_tool", False),
    ],
)
def test_clarify_on_tool_call_first_call(
    tool_id: str | list[str], tool_to_test: str, *, should_raise: bool
) -> None:
    """Test clarify_on_tool_calls on first call with different tool IDs."""
    tool = AdditionTool()
    tool.id = tool_to_test
    args = {"a": 1, "b": 2}
    plan, plan_run = get_test_plan_run()
    step = plan.steps[0]

    hook = clarify_on_tool_calls(tool_id)
    result = hook(tool, args, plan_run, step)

    if should_raise:
        assert isinstance(result, CustomClarification)
        assert result.category == ClarificationCategory.CUSTOM
        assert result.user_guidance is not None
    else:
        assert result is None


def test_clarify_on_tool_call_with_previous_yes_response() -> None:
    """Test clarify_on_tool_calls with a previous 'yes' response."""
    tool = AdditionTool()
    tool.id = "test_tool"
    args = {"a": 1, "b": 2}
    plan, plan_run = get_test_plan_run()
    step = plan.steps[0]

    # Create a previous clarification with 'yes' response
    prev_clarification = CustomClarification(
        name="user_verify",
        plan_run_id=plan_run.id,
        user_guidance=f"Are you happy to proceed with the call to {tool.name}? "
        "Enter 'y' or 'yes' to proceed",
        data={"args": args},
        resolved=True,
        response="yes",
        step=0,
    )
    plan_run.outputs.clarifications = [prev_clarification]

    hook = clarify_on_tool_calls("test_tool")
    result = hook(tool, args, plan_run, step)
    assert result is None


def test_clarify_on_tool_call_with_previous_no_response() -> None:
    """Test clarify_on_tool_calls with a previous 'no' response."""
    tool = AdditionTool()
    tool.id = "test_tool"
    args = {"a": 1, "b": 2}
    plan, plan_run = get_test_plan_run()
    step = plan.steps[0]

    # Create a previous clarification with 'no' response
    prev_clarification = CustomClarification(
        name="user_verify",
        plan_run_id=plan_run.id,
        user_guidance=f"Are you happy to proceed with the call to {tool.name}? "
        "Enter 'y' or 'yes' to proceed",
        data={"args": args},
        resolved=True,
        response="no",
        step=0,
    )
    plan_run.outputs.clarifications = [prev_clarification]

    hook = clarify_on_tool_calls("test_tool")
    with pytest.raises(ToolHardError):
        hook(tool, args, plan_run, step)
