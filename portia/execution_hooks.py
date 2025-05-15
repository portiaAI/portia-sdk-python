"""Execution hooks for customizing the behavior of portia during execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import click
from pydantic import BaseModel, ConfigDict

from portia.clarification import Clarification, ClarificationCategory, CustomClarification
from portia.clarification_handler import ClarificationHandler
from portia.errors import ToolHardError
from portia.execution_agents.output import Output
from portia.plan import Plan, Step
from portia.plan_run import PlanRun
from portia.tool import Tool


class ExecutionHooks(BaseModel):
    """Hooks that can be used to modify or add extra functionality to the run of a plan.

    Hooks can be registered for various execution events:
    - clarification_handler: A handler for clarifications raised during execution
    - before_step_execution: Called before executing each step
    - after_step_execution: Called after executing each step. When there's an error, this is
        called with the error as the output value.
    - before_first_step_execution: Called before executing the first step
    - after_last_step_execution: Called after executing the last step of the plan run. This is not
        called if a clarification is raised, as it is expected that the plan will be resumed after
        the clarification is handled.
    - before_tool_call: Called before the tool is called
    - after_tool_call: Called after the tool is called
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    clarification_handler: ClarificationHandler | None = None
    """Handler for clarifications raised during execution."""

    before_step_execution: Callable[[Plan, PlanRun, Step], None] | None = None
    """Called before executing each step.

    Args:
        plan: The plan being executed
        plan_run: The current plan run
        step: The step about to be executed
    """

    after_step_execution: Callable[[Plan, PlanRun, Step, Output], None] | None = None
    """Called after executing each step.

    When there's an error, this is called with the error as the output value.

    Args:
        plan: The plan being executed
        plan_run: The current plan run
        step: The step that was executed
        output: The output from the step execution
    """

    before_first_step_execution: Callable[[Plan, PlanRun], None] | None = None
    """Called before executing the first step.

    Args:
        plan: The plan being executed
        plan_run: The current plan run
    """

    after_last_step_execution: Callable[[Plan, PlanRun, Output], None] | None = None
    """Called after executing the last step of the plan run.

    This is not called if a clarification is raised, as it is expected that the plan
    will be resumed after the clarification is handled.

    Args:
        plan: The plan that was executed
        plan_run: The completed plan run
        output: The final output from the plan execution
    """

    before_tool_call: (
        Callable[[Tool, dict[str, Any], PlanRun, Step], Clarification | None] | None
    ) = None
    """Called before the tool is called.

    Args:
        tool: The tool about to be called
        args: The args for the tool call
        plan_run: The current plan run
        step: The step being executed

    Returns:
        Clarification | None: A clarification to raise, or None to proceed with the tool call
    """

    after_tool_call: Callable[[Tool, Any, PlanRun, Step], Clarification | None] | None = None
    """Called after the tool is called.

    Args:
        tool: The tool that was called
        output: The output returned from the tool call
        plan_run: The current plan run
        step: The step being executed

    Returns:
        Clarification | None: A clarification to raise, or None to proceed. If a clarification
          is raised, when we later resume the plan, the same step will be executed again
    """


# Example execution hooks


def cli_user_verify_before_tool_call(
    tool: Tool,
    args: dict[str, Any],
    plan_run: PlanRun,
    step: Step,  # noqa: ARG001
) -> Clarification | None:
    """Raise a clarification to check the user is happy with the tool call before proceeding."""
    user_verify_clarification = CustomClarification(
        name="user_verify",
        plan_run_id=plan_run.id,
        user_guidance=f"Are you happy to proceed with the call to {tool.name}? "
        "Enter 'y' or 'yes' to proceed",
        data={"args": args},
    )

    previously_raised_clarification = next(
        (
            c
            for c in plan_run.get_clarifications_for_step()
            if c.user_guidance == user_verify_clarification.user_guidance
            and c.category == ClarificationCategory.CUSTOM
        ),
        None,
    )

    if not previously_raised_clarification or not previously_raised_clarification.resolved:
        return user_verify_clarification

    if str(previously_raised_clarification.response).lower() not in ["y", "yes", "Y", "YES", "Yes"]:
        raise ToolHardError("User rejected tool call to {tool.name} with args {args}")

    return None


def log_step_outputs(plan: Plan, plan_run: PlanRun, step: Step, output: Output) -> None:  # noqa: ARG001
    """Log the output of a step in the plan."""
    click.echo(
        click.style(
            f"Step with task {step.task} using tool {step.tool_id} "
            f"completed with result: {output}",
            fg=87,
        )
    )
