"""Test clarification tool."""

from __future__ import annotations

from portia.clarification import ClarificationCategory, InputClarification
from portia.end_user import EndUser
from portia.execution_agents.clarification_tool import ClarificationTool
from portia.tool import ToolRunContext
from tests.utils import get_test_config, get_test_plan_run


def test_clarification_tool_raises_clarification() -> None:
    """Test that the clarification tool raises a clarification."""
    (plan, plan_run) = get_test_plan_run()
    tool = ClarificationTool(step=plan_run.current_step_index)
    ctx = ToolRunContext(
        end_user=EndUser(external_id="123"),
        plan_run=plan_run,
        plan=plan,
        config=get_test_config(),
        clarifications=[],
    )
    argument_name = "test_argument"

    result = tool.run(ctx, argument_name)

    clarification = InputClarification.model_validate_json(result)

    assert clarification.argument_name == argument_name
    assert clarification.user_guidance == f"Missing Argument: {argument_name}"
    assert clarification.plan_run_id == ctx.plan_run.id
    assert clarification.step == tool.step
    assert clarification.category == ClarificationCategory.INPUT
