"""Plan tests."""

from uuid import UUID

from portia.plan import Plan, PlanContext
from tests.utils import get_test_workflow


def test_plan_serialization() -> None:
    """Test plan can be serialized to string."""
    plan, _ = get_test_workflow()
    assert str(plan) == (
        f"PlanModel(id={plan.id!r},plan_context={plan.plan_context!r}, steps={plan.steps!r}"
    )
    # check we can also serialize to JSON
    plan.model_dump_json()


def test_plan_uuid_assign() -> None:
    """Test plan assign correct UUIDs."""
    plan = Plan(
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
    )
    assert isinstance(plan.id, UUID)

    clarification = Plan(
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
    )
    assert isinstance(clarification.id, UUID)
