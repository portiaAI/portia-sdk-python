"""Plan tests."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from portia.plan import (
    Plan,
    PlanBuilder,
    PlanContext,
    PlanInput,
    PlanUUID,
    ReadOnlyPlan,
    Step,
    UserPlanVote,
    Variable,
)
from tests.utils import get_test_plan_run


def test_plan_serialization() -> None:
    """Test plan can be serialized to string."""
    plan, _ = get_test_plan_run()
    assert str(plan) == (
        f"PlanModel(id={plan.id!r},plan_context={plan.plan_context!r}, steps={plan.steps!r}, "
        f"inputs={plan.plan_inputs!r}"
    )
    # check we can also serialize to JSON
    plan.model_dump_json()


def test_plan_uuid_assign() -> None:
    """Test plan assign correct UUIDs."""
    plan = Plan(
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[Step(task="test task", output="$output")],
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
    for ro_step, orig_step in zip(read_only.steps, original_plan.steps, strict=False):
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


def test_plan_outputs_must_be_unique() -> None:
    """Test that plan outputs must be unique."""
    with pytest.raises(ValidationError, match="Outputs \\+ conditions must be unique"):
        Plan(
            plan_context=PlanContext(query="test query", tool_ids=["tool1"]),
            steps=[
                Step(task="test task", output="$output"),
                Step(task="test task", output="$output"),
            ],
        )


def test_plan_outputs_and_conditions_must_be_unique() -> None:
    """Test that plan outputs and conditions must be unique."""
    with pytest.raises(ValidationError, match="Outputs \\+ conditions must be unique"):
        Plan(
            plan_context=PlanContext(query="test query", tool_ids=["tool1"]),
            steps=[
                Step(task="test task", output="$output", condition="x > 10"),
                Step(task="test task", output="$output", condition="x > 10"),
            ],
        )
    # should not fail if conditions are different
    Plan(
        plan_context=PlanContext(query="test query", tool_ids=["tool1"]),
        steps=[
            Step(task="test task", output="$output", condition="x > 10"),
            Step(task="test task", output="$output", condition="x < 10"),
        ],
    )


def test_pretty_print() -> None:
    """Test pretty print."""
    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=["tool1"]),
        steps=[
            Step(
                task="test task",
                output="$output",
                inputs=[Variable(name="$input", description="test input")],
                condition="x > 10",
            ),
        ],
        plan_inputs=[PlanInput(name="$input", description="test input")],
    )
    output = plan.pretty_print()
    assert isinstance(output, str)


def test_plan_builder_with_plan_input() -> None:
    """Test that plan builder can create plans with plan inputs."""
    plan = (
        PlanBuilder("Process a person's information")
        .step("Process person", "person_processor")
        .plan_input(
            name="$person",
            description="Person's information",
        )
        .build()
    )

    assert len(plan.plan_inputs) == 1
    assert plan.plan_inputs[0].name == "$person"
    assert plan.plan_inputs[0].description == "Person's information"


def test_plan_inputs_must_be_unique() -> None:
    """Test that plan inputs must have unique names."""
    with pytest.raises(ValidationError, match="Plan input names must be unique"):
        Plan(
            plan_context=PlanContext(query="test query", tool_ids=["tool1"]),
            steps=[Step(task="test task", output="$output")],
            plan_inputs=[
                PlanInput(name="$duplicate", description="First input"),
                PlanInput(name="$duplicate", description="Second input with same name"),
            ],
        )


def test_plan_input_equality() -> None:
    """Test equality comparison of PlanInput objects."""
    original_input = PlanInput(name="$test", description="Test input")

    identical_input = PlanInput(name="$test", description="Test input")
    assert original_input == identical_input

    different_descr_input = PlanInput(name="$test", description="Different description")
    assert original_input != different_descr_input

    different_name_input = PlanInput(name="$different", description="Test input")
    assert original_input != different_name_input

    # Test inequality with different types
    assert original_input != "not a plan input"
    assert original_input != 42


def test_plan_with_voting_metadata() -> None:
    """Test that Plan can be created with voting metadata."""
    now = datetime.now(UTC)
    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=["tool1"]),
        steps=[Step(task="test task", output="$output")],
        upvotes=10,
        downvotes=2,
        created_by="user123",
        created_at=now,
        num_runs=5,
        is_upvoted=True,
    )

    assert plan.upvotes == 10
    assert plan.downvotes == 2
    assert plan.created_by == "user123"
    assert plan.created_at == now
    assert plan.num_runs == 5
    assert plan.is_upvoted is True


def test_plan_voting_metadata_defaults() -> None:
    """Test that Plan voting metadata fields have proper defaults."""
    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=["tool1"]),
        steps=[Step(task="test task", output="$output")],
    )

    assert plan.upvotes == 0
    assert plan.downvotes == 0
    assert plan.created_by is None
    assert plan.created_at is None
    assert plan.num_runs == 0
    assert plan.is_upvoted is False


def test_plan_from_response_with_voting_fields() -> None:
    """Test Plan.from_response handles voting metadata fields."""
    plan_uuid = str(PlanUUID())
    response_json = {
        "id": plan_uuid,
        "query": "test query",
        "tool_ids": ["tool1"],
        "steps": [{"task": "test task", "output": "$output", "inputs": []}],
        "plan_inputs": [],
        "upvotes": 15,
        "downvotes": 3,
        "created_by": "user456",
        "created_at": "2025-01-15T10:30:00Z",
        "num_runs": 8,
        "is_upvoted": True,
    }

    plan = Plan.from_response(response_json)

    assert plan.upvotes == 15
    assert plan.downvotes == 3
    assert plan.created_by == "user456"
    assert plan.created_at is not None
    assert plan.num_runs == 8
    assert plan.is_upvoted is True


def test_plan_from_response_without_voting_fields() -> None:
    """Test Plan.from_response handles missing voting metadata fields."""
    plan_uuid = str(PlanUUID())
    response_json = {
        "id": plan_uuid,
        "query": "test query",
        "tool_ids": ["tool1"],
        "steps": [{"task": "test task", "output": "$output", "inputs": []}],
        "plan_inputs": [],
    }

    plan = Plan.from_response(response_json)

    assert plan.upvotes == 0
    assert plan.downvotes == 0
    assert plan.created_by is None
    assert plan.created_at is None
    assert plan.num_runs == 0
    assert plan.is_upvoted is False


def test_user_plan_vote_creation() -> None:
    """Test UserPlanVote can be created with valid data."""
    plan_id = PlanUUID()
    vote = UserPlanVote(
        user_id="user123",
        plan_id=plan_id,
        vote_status="up",
    )

    assert vote.user_id == "user123"
    assert vote.plan_id == plan_id
    assert vote.vote_status == "up"


def test_user_plan_vote_validation() -> None:
    """Test UserPlanVote validates vote_status."""
    plan_id = PlanUUID()

    # Valid vote statuses
    UserPlanVote(user_id="user123", plan_id=plan_id, vote_status="up")
    UserPlanVote(user_id="user123", plan_id=plan_id, vote_status="down")

    # Invalid vote status
    with pytest.raises(ValidationError, match="vote_status must be either 'up' or 'down'"):
        UserPlanVote(user_id="user123", plan_id=plan_id, vote_status="invalid")


def test_readonly_plan_preserves_voting_metadata() -> None:
    """Test that ReadOnlyPlan preserves voting metadata."""
    now = datetime.now(UTC)
    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=["tool1"]),
        steps=[Step(task="test task", output="$output")],
        upvotes=20,
        downvotes=5,
        created_by="user789",
        created_at=now,
        num_runs=12,
        is_upvoted=True,
    )

    read_only = ReadOnlyPlan.from_plan(plan)

    assert read_only.upvotes == plan.upvotes
    assert read_only.downvotes == plan.downvotes
    assert read_only.created_by == plan.created_by
    assert read_only.created_at == plan.created_at
    assert read_only.num_runs == plan.num_runs
    assert read_only.is_upvoted == plan.is_upvoted
