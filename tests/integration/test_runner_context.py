"""Tests for execution context."""

from __future__ import annotations

from portia.plan import Plan, PlanContext, Step
from portia.plan_run import PlanRun
from portia.tool import Tool, ToolRunContext


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
    return plan, PlanRun(plan_id=plan.id, current_step_index=0, end_user_id="123")
