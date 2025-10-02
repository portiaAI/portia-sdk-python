"""Execution hooks for customizing the behavior of portia during execution."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from pydantic import BaseModel, ConfigDict

from portia.builder.plan_v2 import PlanV2
from portia.builder.step_v2 import StepV2
from portia.clarification import (
    Clarification,
    ClarificationCategory,
    UserVerificationClarification,
)
from portia.clarification_handler import ClarificationHandler
from portia.common import PortiaEnum
from portia.errors import ToolHardError
from portia.execution_agents.output import Output
from portia.logger import logger
from portia.plan_run import PlanRunV2
from portia.tool import Tool


class BeforeStepExecutionOutcome(PortiaEnum):
    """The Outcome of the before step execution hook."""

    # Continue with the step execution
    CONTINUE = "CONTINUE"
    # Skip the step execution
    SKIP = "SKIP"


class ExecutionHooks(BaseModel):
    """Hooks that can be used to modify or add extra functionality to the run of a plan.

    Hooks can be registered for various execution events:
    - clarification_handler: A handler for clarifications raised during execution
    - before_step_execution: Called before executing each step
    - after_step_execution: Called after executing each step. When there's an error, this is
        called with the error as the output value.
    - before_plan_run: Called before executing the first step of the plan run.
    - after_plan_run: Called after executing the plan run. This is not called if a clarification
        is raised, as it is expected that the plan will be resumed after the clarification is
        handled.
    - before_tool_call: Called before the tool is called
    - after_tool_call: Called after the tool is called
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    clarification_handler: ClarificationHandler | None = None
    """Handler for clarifications raised during execution."""

    before_step_execution: Callable[[PlanV2, PlanRunV2, StepV2], BeforeStepExecutionOutcome] | None = None
    """Called before executing each step.

    Args:
        plan: The PlanV2 being executed
        plan_run: The current PlanRunV2
        step: The StepV2 about to be executed

    Returns:
        BeforeStepExecutionOutcome | None: Whether to continue with the step execution or skip it.
            If None is returned, the default behaviour is to continue with the step execution.
    """

    after_step_execution: Callable[[PlanV2, PlanRunV2, StepV2, Output], None] | None = None
    """Called after executing each step.

    When there's an error, this is called with the error as the output value.

    Args:
        plan: The PlanV2 being executed
        plan_run: The current PlanRunV2
        step: The StepV2 that was executed
        output: The output from the step execution
    """

    before_plan_run: Callable[[PlanV2, PlanRunV2], None] | None = None
    """Called before executing the first step of the plan run.

    Args:
        plan: The PlanV2 being executed
        plan_run: The current PlanRunV2
    """

    after_plan_run: Callable[[PlanV2, PlanRunV2, Output], None] | None = None
    """Called after executing the plan run.

    This is not called if a clarification is raised, as it is expected that the plan
    will be resumed after the clarification is handled.

    Args:
        plan: The PlanV2 that was executed
        plan_run: The completed PlanRunV2
        output: The final output from the plan execution
    """

    before_tool_call: (
        Callable[[Tool, dict[str, Any], PlanRunV2, StepV2], Clarification | None] | None
    ) = None
    """Called before the tool is called.

    Args:
        tool: The tool about to be called
        args: The args for the tool call. These are mutable and so can be modified in place as
          required.
        plan_run: The current PlanRunV2
        step: The StepV2 being executed

    Returns:
        Clarification | None: A clarification to raise, or None to proceed with the tool call
    """

    after_tool_call: Callable[[Tool, Any, PlanRunV2, StepV2], Clarification | None] | None = None
    """Called after the tool is called.

    Args:
        tool: The tool that was called
        output: The output returned from the tool call
        plan_run: The current PlanRunV2
        step: The StepV2 being executed

    Returns:
        Clarification | None: A clarification to raise, or None to proceed. If a clarification
          is raised, when we later resume the plan, the same step will be executed again
    """


# Example execution hooks


def clarify_on_all_tool_calls(
    tool: Tool,
    args: dict[str, Any],
    plan_run: PlanRunV2,
    step: StepV2,
) -> Clarification | None:
    """Raise a clarification to check the user is happy with all tool calls before proceeding.

    Example usage:
        portia = Portia(
            execution_hooks=ExecutionHooks(
                before_tool_call=clarify_on_all_tool_calls,
            )
        )
    """
    return _clarify_on_tool_call_hook(tool, args, plan_run, step, tool_ids=None)


def clarify_on_tool_calls(
    tool: str | Tool | list[str] | list[Tool],
) -> Callable[[Tool, dict[str, Any], PlanRunV2, StepV2], Clarification | None]:
    """Return a hook that raises a clarification before calls to the specified tool.

    Args:
        tool: The tool or tools to raise a clarification for before running

    Example usage:
        portia = Portia(
            execution_hooks=ExecutionHooks(
                before_tool_call=clarify_on_tool_calls("my_tool_id"),
            )
        )
        # Or with Tool objects:
        portia = Portia(
            execution_hooks=ExecutionHooks(
                before_tool_call=clarify_on_tool_calls([tool1, tool2]),
            )
        )

    """
    if isinstance(tool, Tool):
        tool_ids = [tool.id]
    elif isinstance(tool, str):
        tool_ids = [tool]
    else:
        tool_ids = [t.id if isinstance(t, Tool) else t for t in tool]

    return partial(_clarify_on_tool_call_hook, tool_ids=tool_ids)


def _clarify_on_tool_call_hook(
    tool: Tool,
    args: dict[str, Any],
    plan_run: PlanRunV2,
    step: StepV2,  # noqa: ARG001
    tool_ids: list[str] | None,
) -> Clarification | None:
    """Raise a clarification to check the user is happy with all tool calls before proceeding."""
    if tool_ids and tool.id not in tool_ids:
        return None

    previous_clarification = plan_run.get_clarification_for_step(
        ClarificationCategory.USER_VERIFICATION
    )
    serialised_args = (
        ", ".join([f"{k}={v}" for k, v in args.items()]).replace("{", "[").replace("}", "]")
    )

    if not previous_clarification or not previous_clarification.resolved:
        return UserVerificationClarification(
            plan_run_id=plan_run.id,
            user_guidance=f"Are you happy to proceed with the call to {tool.name} with args "
            f"{serialised_args}? Enter 'y' or 'yes' to proceed",
            source="User verification tool hook",
        )

    if (
        previous_clarification.category == ClarificationCategory.USER_VERIFICATION
        and previous_clarification.response is False
    ):
        raise ToolHardError(f"User rejected tool call to {tool.name} with args {args}")

    return None


def log_step_outputs(plan: PlanV2, plan_run: PlanRunV2, step: StepV2, output: Output) -> None:  # noqa: ARG001
    """Log the output of a step in the plan."""
    logger().info(
        f"Step {step.step_name} completed with result: {output}"
    )
