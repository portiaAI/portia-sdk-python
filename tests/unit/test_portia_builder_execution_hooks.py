from unittest.mock import MagicMock

from portia.builder.plan_builder_v2 import PlanBuilderV2
from portia.execution_hooks import BeforeStepExecutionOutcome, ExecutionHooks
from portia.plan_run import PlanRunState
from portia.portia import Portia


def test_run_builder_plan_execution_hooks(portia: Portia) -> None:
    """Test that execution hooks are called when running a PlanV2."""
    execution_hooks = ExecutionHooks(
        before_plan_run=MagicMock(),
        before_step_execution=MagicMock(),
        after_step_execution=MagicMock(),
        after_plan_run=MagicMock(),
    )
    portia.execution_hooks = execution_hooks
    plan = (
        PlanBuilderV2("Test execution hooks")
        .function_step(function=lambda: "Step 1 result")
        .function_step(function=lambda: "Step 2 result")
        .build()
    )

    plan_run = portia.run_plan(plan)

    assert plan_run.state == PlanRunState.COMPLETE
    assert execution_hooks.before_plan_run.call_count == 1
    assert execution_hooks.before_step_execution.call_count == 2
    assert execution_hooks.after_step_execution.call_count == 2
    assert execution_hooks.after_plan_run.call_count == 1


def test_run_builder_plan_execution_hooks_with_skip(portia: Portia) -> None:
    """Test that before_step_execution can skip steps for PlanV2."""

    def before_step_execution(plan, plan_run, step):  # noqa: ANN001
        return (
            BeforeStepExecutionOutcome.SKIP
            if step.output == "$s0_output"
            else BeforeStepExecutionOutcome.CONTINUE
        )

    execution_hooks = ExecutionHooks(
        before_step_execution=before_step_execution,
        after_step_execution=MagicMock(),
    )
    portia.execution_hooks = execution_hooks
    plan = (
        PlanBuilderV2("Test execution hooks with skip")
        .function_step(function=lambda: "Step 1 result")
        .function_step(function=lambda: "Step 2 result")
        .build()
    )

    plan_run = portia.run_plan(plan)

    assert plan_run.state == PlanRunState.COMPLETE
    assert execution_hooks.after_step_execution.call_count == 1
