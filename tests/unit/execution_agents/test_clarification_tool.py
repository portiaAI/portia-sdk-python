"""Test clarification tool."""

from __future__ import annotations

from portia.clarification import ClarificationCategory, InputClarification
from portia.end_user import EndUser
from portia.execution_agents.clarification_tool import ClarificationTool
from portia.prefixed_uuid import PlanRunUUID
from portia.tool import ToolRunContext
from tests.utils import get_test_config


def test_clarification_tool_raises_clarification() -> None:
    """Test that the clarification tool raises a clarification correctly."""
    tool = ClarificationTool(step=1)
    ctx = ToolRunContext(
        end_user=EndUser(external_id="123"),
        plan_run_id=PlanRunUUID(),
        config=get_test_config(),
        clarifications=[],
    )
    argument_name = "test_argument"

    result = tool.run(ctx, argument_name)

    clarification = InputClarification.model_validate_json(result)

    assert clarification.argument_name == argument_name
    assert clarification.user_guidance == f"Missing Argument: {argument_name}"
    assert clarification.plan_run_id == ctx.plan_run_id
    assert clarification.step == tool.step
    assert clarification.category == ClarificationCategory.INPUT
