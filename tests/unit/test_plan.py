"""Plan tests."""

import pytest
from pydantic import ValidationError

from portia.plan import Plan, PlanContext, PlanUUID, ReadOnlyPlan, Step, Variable
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
    assert isinstance(plan.id, PlanUUID)


def test_read_only_plan_immutable() -> None:
    """Test immutability of ReadOnlyPlan."""
    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=[]),
        steps=[
            Step(task="test task", output="$output"),
        ],
    )
    read_only = ReadOnlyPlan.from_plan(plan)

    with pytest.raises(ValidationError):
        read_only.steps = []

    with pytest.raises(ValidationError):
        read_only.plan_context = PlanContext(query="new query", tool_ids=[])


def test_read_only_plan_preserves_data() -> None:
    """Test that ReadOnlyPlan preserves all data from original Plan."""
    original_plan = Plan(
        plan_context=PlanContext(
            query="What's the weather?",
            tool_ids=["weather_tool"],
        ),
        steps=[
            Step(task="Get weather", output="$weather"),
            Step(task="Format response", output="$response"),
        ],
    )

    read_only = ReadOnlyPlan.from_plan(original_plan)

    # Verify all data is preserved
    assert read_only.id == original_plan.id
    assert read_only.plan_context.query == original_plan.plan_context.query
    assert read_only.plan_context.tool_ids == original_plan.plan_context.tool_ids
    assert len(read_only.steps) == len(original_plan.steps)
    for ro_step, orig_step in zip(read_only.steps, original_plan.steps):
        assert ro_step.task == orig_step.task
        assert ro_step.output == orig_step.output


def test_read_only_plan_serialization() -> None:
    """Test that ReadOnlyPlan can be serialized and deserialized."""
    original_plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=["tool1"]),
        steps=[Step(task="test task", output="$output")],
    )
    read_only = ReadOnlyPlan.from_plan(original_plan)

    json_str = read_only.model_dump_json()

    deserialized = ReadOnlyPlan.model_validate_json(json_str)

    # Verify data is preserved through serialization
    assert deserialized.id == read_only.id
    assert deserialized.plan_context.query == read_only.plan_context.query
    assert deserialized.plan_context.tool_ids == read_only.plan_context.tool_ids
    assert len(deserialized.steps) == len(read_only.steps)
    assert deserialized.steps[0].task == read_only.steps[0].task
    assert deserialized.steps[0].output == read_only.steps[0].output


def test_plan_builder_syntax() -> None:
    """Test that the plan builder syntax works."""
    plan_builder = (
        Plan.context("Find the best offers for a flight from London to New York", ["flight_search"])
        .step(
            "Search for flights",
            "$flights",
            tool_id="flight_search",
        )
        .variable(
            "$flights",
        )
    )
    base_plan = Plan(
        plan_context=PlanContext(
            query="Find the best offers for a flight from London to New York",
            tool_ids=["flight_search"],
        ),
        steps=[
            Step(
                task="Search for flights",
                output="$flights",
                tool_id="flight_search",
                inputs=[Variable(name="$flights", description="")],
            ),
        ],
    )
    assert plan_builder.model_dump(exclude={"id"}) == base_plan.model_dump(exclude={"id"})


def test_plan_variable_no_steps() -> None:
    """Test that the plan variable function raises an error if there are no steps."""
    plan = Plan.context(
        "Find the best offers for a flight from London to New York", ["flight_search"],
    )
    with pytest.raises(ValueError, match="No steps in the plan"):
        plan.variable("$flights")
