"""Tests for execution context."""

from __future__ import annotations

from portia.config import StorageClass, default_config
from portia.execution_context import ExecutionContext, execution_context
from portia.plan import Plan, PlanContext, Step
from portia.plan_run import PlanRun, PlanRunState
from portia.portia import Portia
from portia.tool import Tool, ToolRunContext
from portia.tool_registry import ToolRegistry


class ExecutionContextTrackerTool(Tool):
    """Tracks Execution Context."""

    id: str = "execution_tracker_tool"
    name: str = "Execution Tracker Tool"
    description: str = "Tracks tool execution context"
    output_schema: tuple[str, str] = (
        "None",
        "Nothing",
    )
    tool_context: ToolRunContext | None = None

    def run(
        self,
        ctx: ToolRunContext,
    ) -> None:
        """Save the context."""
        self.tool_context = ctx


def get_test_plan_run() -> tuple[Plan, PlanRun]:
    """Return test plan_run."""
    step1 = Step(
        task="Save Context",
        inputs=[],
        output="$ctx",
        tool_id="execution_tracker_tool",
    )
    plan = Plan(
        plan_context=PlanContext(
            query="Add 1 + 2",
            tool_ids=["add_tool"],
        ),
        steps=[step1],
    )
    return plan, PlanRun(plan_id=plan.id, current_step_index=0)


def test_portia_no_execution_context_new() -> None:
    """Test running a query."""
    tool = ExecutionContextTrackerTool()
    tool_registry = ToolRegistry([tool])
    portia = Portia(tools=tool_registry, config=default_config(storage_class=StorageClass.MEMORY))
    (plan, plan_run) = get_test_plan_run()
    portia.storage.save_plan(plan)
    plan_run = portia.resume(plan_run)

    assert plan_run.state == PlanRunState.COMPLETE
    assert tool.tool_context
    assert tool.tool_context.plan_run_id == plan_run.id


def test_portia_no_execution_context_existing() -> None:
    """Test running a query."""
    tool = ExecutionContextTrackerTool()
    tool_registry = ToolRegistry([tool])
    portia = Portia(tools=tool_registry, config=default_config(storage_class=StorageClass.MEMORY))
    (plan, plan_run) = get_test_plan_run()
    plan_run.execution_context = ExecutionContext(end_user_id="123")
    portia.storage.save_plan(plan)
    plan_run = portia.resume(plan_run)

    assert plan_run.state == PlanRunState.COMPLETE
    assert tool.tool_context
    assert tool.tool_context.plan_run_id == plan_run.id
    assert tool.tool_context.execution_context.end_user_id == "123"


def test_portia_with_execution_context_new() -> None:
    """Test running a query."""
    tool = ExecutionContextTrackerTool()
    tool_registry = ToolRegistry([tool])
    portia = Portia(tools=tool_registry, config=default_config(storage_class=StorageClass.MEMORY))
    (plan, plan_run) = get_test_plan_run()
    portia.storage.save_plan(plan)

    with execution_context(end_user_id="123"):
        plan_run = portia.resume(plan_run)

    assert plan_run.state == PlanRunState.COMPLETE
    assert tool.tool_context
    assert tool.tool_context.plan_run_id == plan_run.id
    assert tool.tool_context.execution_context.end_user_id == "123"


def test_portia_with_execution_context_existing() -> None:
    """Test running a query."""
    tool = ExecutionContextTrackerTool()
    tool_registry = ToolRegistry([tool])
    portia = Portia(tools=tool_registry, config=default_config(storage_class=StorageClass.MEMORY))
    (plan, plan_run) = get_test_plan_run()
    plan_run.execution_context = ExecutionContext()
    portia.storage.save_plan(plan)

    with execution_context(end_user_id="123"):
        plan_run = portia.resume(plan_run)

    assert plan_run.state == PlanRunState.COMPLETE
    assert tool.tool_context
    assert tool.tool_context.plan_run_id == plan_run.id
    assert tool.tool_context.execution_context.end_user_id == "123"
