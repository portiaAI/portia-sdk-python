"""Tests for execution context."""

from __future__ import annotations

from portia.config import Config
from portia.context import ExecutionContext, execution_context
from portia.plan import Plan, PlanContext, Step
from portia.runner import Runner
from portia.tool import Tool
from portia.tool_registry import InMemoryToolRegistry
from portia.workflow import Workflow, WorkflowState


class ExecutionContextTrackerTool(Tool):
    """Tracks Execution Context."""

    id: str = "execution_tracker_tool"
    name: str = "Execution Tracker Tool"
    description: str = "Tracks execution context"
    output_schema: tuple[str, str] = (
        "None",
        "Nothing",
    )
    execution_context: ExecutionContext | None = None

    def run(
        self,
        ctx: ExecutionContext,
    ) -> None:
        """Save the context."""
        print("tool ctx:", ctx)
        self.execution_context = ctx


def get_test_workflow() -> tuple[Plan, Workflow]:
    """Return test workflow."""
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
    return plan, Workflow(plan_id=plan.id, current_step_index=0)


def test_runner_no_execution_context_new() -> None:
    """Test running a query using the Runner."""
    config = Config.from_default()
    tool = ExecutionContextTrackerTool()
    tool_registry = InMemoryToolRegistry.from_local_tools([tool])
    runner = Runner(config=config, tool_registry=tool_registry)
    (plan, workflow) = get_test_workflow()
    runner.storage.save_plan(plan)
    workflow = runner.execute_workflow(workflow)

    assert workflow.state == WorkflowState.COMPLETE
    assert tool.execution_context
    assert tool.execution_context.workflow_id == str(workflow.id)


def test_runner_no_execution_context_existing() -> None:
    """Test running a query using the Runner."""
    config = Config.from_default()
    tool = ExecutionContextTrackerTool()
    tool_registry = InMemoryToolRegistry.from_local_tools([tool])
    runner = Runner(config=config, tool_registry=tool_registry)
    (plan, workflow) = get_test_workflow()
    workflow.execution_context = ExecutionContext(workflow_id=str(workflow.id), end_user_id="123")
    runner.storage.save_plan(plan)
    workflow = runner.execute_workflow(workflow)

    assert workflow.state == WorkflowState.COMPLETE
    assert tool.execution_context
    assert tool.execution_context.workflow_id == str(workflow.id)
    assert tool.execution_context.end_user_id == "123"


def test_runner_with_execution_context_new() -> None:
    """Test running a query using the Runner."""
    config = Config.from_default()
    tool = ExecutionContextTrackerTool()
    tool_registry = InMemoryToolRegistry.from_local_tools([tool])
    runner = Runner(config=config, tool_registry=tool_registry)
    (plan, workflow) = get_test_workflow()
    runner.storage.save_plan(plan)

    with execution_context(end_user_id="123"):
        workflow = runner.execute_workflow(workflow)

    assert workflow.state == WorkflowState.COMPLETE
    assert tool.execution_context
    assert tool.execution_context.workflow_id == str(workflow.id)
    assert tool.execution_context.end_user_id == "123"


def test_runner_with_execution_context_existing() -> None:
    """Test running a query using the Runner."""
    config = Config.from_default()
    tool = ExecutionContextTrackerTool()
    tool_registry = InMemoryToolRegistry.from_local_tools([tool])
    runner = Runner(config=config, tool_registry=tool_registry)
    (plan, workflow) = get_test_workflow()
    workflow.execution_context = ExecutionContext(workflow_id=str(workflow.id))
    runner.storage.save_plan(plan)

    with execution_context(end_user_id="123"):
        workflow = runner.execute_workflow(workflow)

    assert workflow.state == WorkflowState.COMPLETE
    assert tool.execution_context
    assert tool.execution_context.workflow_id == str(workflow.id)
    assert tool.execution_context.end_user_id == "123"
