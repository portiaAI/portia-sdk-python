"""Test the Portia class with PlanV2."""

from __future__ import annotations

import asyncio
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from portia.builder.plan_builder_v2 import PlanBuilderV2
from portia.builder.plan_v2 import PlanV2
from portia.builder.reference import Input
from portia.execution_hooks import BeforeStepExecutionOutcome, ExecutionHooks
from portia.plan import Plan, PlanContext, PlanInput, Step
from portia.plan_run import PlanRun, PlanRunState
from portia.portia import Portia


def _build_addition_plan() -> PlanV2:
    """Build a simple addition plan using the builder."""
    return (
        PlanBuilderV2("Add two numbers")
        .input(name="num_a", description="First number to add")
        .input(name="num_b", description="Second number to add")
        .invoke_tool_step(
            step_name="Add numbers",
            tool="add_tool",
            args={"a": Input("num_a"), "b": Input("num_b")},
        )
        .build()
    )


def test_portia_run_builder_plan(portia: Portia) -> None:
    """Test that run_builder_plan calls internal helpers correctly."""
    plan = PlanV2(steps=[])
    end_user = portia.initialize_end_user()

    mock_plan_run = MagicMock()
    mock_resumed_plan_run = MagicMock()

    with (
        mock.patch.object(
            portia, "_aget_plan_run_from_plan", new_callable=AsyncMock
        ) as mock_get_plan_run,
        mock.patch.object(portia, "resume_builder_plan", new_callable=AsyncMock) as mock_resume,
    ):
        mock_get_plan_run.return_value = mock_plan_run
        mock_resume.return_value = mock_resumed_plan_run

        result = asyncio.run(portia.run_builder_plan(plan=plan, end_user=end_user))

    mock_get_plan_run.assert_awaited_once()
    legacy_plan = mock_get_plan_run.call_args.args[0]
    assert isinstance(legacy_plan, Plan)
    mock_resume.assert_awaited_once_with(
        plan, mock_plan_run, end_user=end_user, legacy_plan=legacy_plan
    )
    assert result == mock_resumed_plan_run


def test_portia_resume_builder_plan_after_interruption(portia: Portia) -> None:
    """Test resuming a PlanV2 run after interruption."""
    plan = PlanV2(steps=[])
    end_user = portia.initialize_end_user()
    legacy_plan = plan.to_legacy_plan(
        PlanContext(
            query=plan.label,
            tool_ids=[tool.id for tool in portia.tool_registry.get_tools()],
        )
    )
    plan_run = asyncio.run(portia._aget_plan_run_from_plan(legacy_plan, end_user, None))

    plan_run.state = PlanRunState.IN_PROGRESS
    plan_run.current_step_index = 1

    plan_run = asyncio.run(portia.resume_builder_plan(plan, plan_run))

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.current_step_index == 1


def test_portia_set_run_state_to_fail_if_keyboard_interrupt_when_resume_builder_plan(
    portia: Portia,
) -> None:
    """Test run state set to FAILED if a KeyboardInterrupt is raised."""
    plan = PlanV2(steps=[])
    end_user = portia.initialize_end_user()
    legacy_plan = plan.to_legacy_plan(
        PlanContext(
            query=plan.label,
            tool_ids=[tool.id for tool in portia.tool_registry.get_tools()],
        )
    )
    plan_run = asyncio.run(portia._aget_plan_run_from_plan(legacy_plan, end_user, None))
    plan_run.state = PlanRunState.IN_PROGRESS

    with mock.patch.object(portia, "_execute_builder_plan", side_effect=KeyboardInterrupt):
        asyncio.run(portia.resume_builder_plan(plan, plan_run))

    assert plan_run.state == PlanRunState.FAILED


def test_portia_resume_builder_plan_invalid_state(portia: Portia) -> None:
    """Test resuming PlanV2 run with an invalid state."""
    plan = PlanV2(steps=[])
    end_user = portia.initialize_end_user()
    legacy_plan = plan.to_legacy_plan(
        PlanContext(
            query=plan.label,
            tool_ids=[tool.id for tool in portia.tool_registry.get_tools()],
        )
    )
    plan_run = asyncio.run(portia._aget_plan_run_from_plan(legacy_plan, end_user, None))
    plan_run.state = PlanRunState.COMPLETE

    result = asyncio.run(portia.resume_builder_plan(plan, plan_run))

    assert result is plan_run
    assert result.state == PlanRunState.COMPLETE


@pytest.mark.parametrize(
    "plan_run_inputs",
    [
        [PlanInput(name="num_a", value=1), PlanInput(name="num_b", value=2)],
        [{"name": "num_a", "value": 1}, {"name": "num_b", "value": 2}],
        {"num_a": 1, "num_b": 2},
        [{"incorrect_key": "num_a", "error": "Error"}],
        "error",
    ],
)
def test_portia_run_builder_plan_with_plan_run_inputs(
    portia: Portia,
    plan_run_inputs: list[PlanInput] | list[dict[str, int]] | dict[str, int] | str,
) -> None:
    """Test that run_builder_plan handles plan inputs in different formats."""
    plan = _build_addition_plan()
    end_user = portia.initialize_end_user()

    if plan_run_inputs == "error" or (
        isinstance(plan_run_inputs, list)
        and isinstance(plan_run_inputs[0], dict)
        and "error" in plan_run_inputs[0]
    ):
        with pytest.raises(ValueError):  # noqa: PT011
            asyncio.run(portia.run_builder_plan(plan, end_user, plan_run_inputs=plan_run_inputs))
        return

    plan_run = asyncio.run(portia.run_builder_plan(plan, end_user, plan_run_inputs=plan_run_inputs))

    assert plan_run.plan_id == plan.id
    assert len(plan_run.plan_run_inputs) == 2
    assert plan_run.plan_run_inputs["num_a"].get_value() == 1
    assert plan_run.plan_run_inputs["num_b"].get_value() == 2
    assert plan_run.outputs.final_output is not None
    assert plan_run.outputs.final_output.get_value() == 3


def test_portia_run_builder_plan_with_missing_inputs(portia: Portia) -> None:
    """Test that run_builder_plan raises error when required inputs are missing."""
    plan = _build_addition_plan()
    end_user = portia.initialize_end_user()

    with pytest.raises(ValueError):  # noqa: PT011
        asyncio.run(portia.run_builder_plan(plan, end_user, plan_run_inputs=[]))

    with pytest.raises(ValueError):  # noqa: PT011
        asyncio.run(
            portia.run_builder_plan(
                plan,
                end_user,
                plan_run_inputs=[PlanInput(name="num_a", value=1)],
            )
        )

    plan_run = asyncio.run(
        portia.run_builder_plan(
            plan,
            end_user,
            plan_run_inputs=[
                PlanInput(name="num_a", value=1),
                PlanInput(name="num_b", value=2),
            ],
        )
    )
    assert plan_run.state == PlanRunState.COMPLETE


def test_portia_run_builder_plan_with_extra_input_when_expecting_none(
    portia: Portia,
) -> None:
    """Test that run_builder_plan ignores inputs when none are expected."""
    plan = PlanV2(steps=[], plan_inputs=[])
    end_user = portia.initialize_end_user()
    extra_input = PlanInput(name="extra", value="value")

    plan_run = asyncio.run(portia.run_builder_plan(plan, end_user, plan_run_inputs=[extra_input]))

    assert plan_run.plan_run_inputs == {}


def test_portia_run_builder_plan_with_additional_extra_input(portia: Portia) -> None:
    """Test that run_builder_plan ignores unknown inputs."""
    expected_input = PlanInput(name="expected", description="Expected input")
    plan = PlanV2(steps=[], plan_inputs=[expected_input])
    end_user = portia.initialize_end_user()
    unknown_input = PlanInput(name="unknown", value="unknown_value")

    plan_run = asyncio.run(
        portia.run_builder_plan(
            plan,
            end_user,
            plan_run_inputs=[
                PlanInput(name="expected", value="expected_value"),
                unknown_input,
            ],
        )
    )

    assert len(plan_run.plan_run_inputs) == 1
    assert plan_run.plan_run_inputs["expected"].get_value() == "expected_value"


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
    assert execution_hooks.before_plan_run.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
    assert execution_hooks.before_step_execution.call_count == 2  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
    assert execution_hooks.after_step_execution.call_count == 2  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
    assert execution_hooks.after_plan_run.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]


def test_run_builder_plan_execution_hooks_with_skip(portia: Portia) -> None:
    """Test that before_step_execution can skip steps for PlanV2."""

    def before_step_execution(plan: Plan, plan_run: PlanRun, step: Step):  # noqa: ANN202, ARG001
        return (
            BeforeStepExecutionOutcome.SKIP
            if step.output == "$step_0_output"
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
    assert execution_hooks.after_step_execution.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
