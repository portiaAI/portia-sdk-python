"""Tests for Run primitives."""

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from portia.clarification import Clarification, ClarificationCategory, InputClarification
from portia.errors import ToolHardError, ToolSoftError
from portia.execution_agents.output import LocalDataValue
from portia.plan import PlanUUID, ReadOnlyStep, Step
from portia.plan_run import (
    PlanRun,
    PlanRunOutputs,
    PlanRunState,
    PlanRunV2,
    ReadOnlyPlanRun,
    migrate_plan_run_to_v2,
    migrate_v2_to_plan_run,
)
from portia.prefixed_uuid import PlanRunUUID


@pytest.fixture
def mock_clarification() -> InputClarification:
    """Create a mock clarification for testing."""
    return InputClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="test",
        resolved=False,
        argument_name="test",
        source="Test plan run",
    )


@pytest.fixture
def plan_run(mock_clarification: InputClarification) -> PlanRun:
    """Create PlanRun instance for testing."""
    return PlanRun(
        plan_id=PlanUUID(),
        current_step_index=1,
        state=PlanRunState.IN_PROGRESS,
        end_user_id="test123",
        outputs=PlanRunOutputs(
            clarifications=[mock_clarification],
            step_outputs={"step1": LocalDataValue(value="Test output")},
        ),
    )


def test_run_initialization() -> None:
    """Test initialization of PlanRun instance."""
    plan_id = PlanUUID()
    plan_run_inputs = {"$input1": LocalDataValue(value="test_input_value")}
    plan_run = PlanRun(
        plan_id=plan_id,
        end_user_id="test123",
        plan_run_inputs=plan_run_inputs,
    )

    assert plan_run.id is not None
    assert plan_run.plan_id == plan_id
    assert isinstance(plan_run.plan_id.uuid, UUID)
    assert plan_run.current_step_index == 0
    assert plan_run.outputs.clarifications == []
    assert plan_run.state == PlanRunState.NOT_STARTED
    assert plan_run.outputs.step_outputs == {}
    assert len(plan_run.plan_run_inputs) == 1
    assert plan_run.plan_run_inputs["$input1"].get_value() == "test_input_value"
    assert plan_run.get_potential_step_inputs() == plan_run_inputs


def test_run_get_outstanding_clarifications(
    plan_run: PlanRun,
    mock_clarification: Clarification,
) -> None:
    """Test get_outstanding_clarifications method."""
    outstanding = plan_run.get_outstanding_clarifications()

    assert len(outstanding) == 1
    assert outstanding[0] == mock_clarification


def test_run_get_outstanding_clarifications_none() -> None:
    """Test get_outstanding_clarifications when no clarifications are outstanding."""
    plan_run = PlanRun(
        plan_id=PlanUUID(),
        outputs=PlanRunOutputs(clarifications=[]),
        end_user_id="test123",
    )

    assert plan_run.get_outstanding_clarifications() == []


def test_run_state_enum() -> None:
    """Test the RunState enum values."""
    assert PlanRunState.NOT_STARTED == "NOT_STARTED"
    assert PlanRunState.IN_PROGRESS == "IN_PROGRESS"
    assert PlanRunState.COMPLETE == "COMPLETE"
    assert PlanRunState.NEED_CLARIFICATION == "NEED_CLARIFICATION"
    assert PlanRunState.FAILED == "FAILED"


def test_read_only_run_immutable() -> None:
    """Test immutability of plan_run."""
    plan_run = PlanRun(
        plan_id=PlanUUID(uuid=uuid4()),
        end_user_id="test123",
    )
    read_only = ReadOnlyPlanRun.from_plan_run(plan_run)

    with pytest.raises(ValidationError):
        read_only.state = PlanRunState.IN_PROGRESS


def test_read_only_step_immutable() -> None:
    """Test immutability of step."""
    step = Step(task="add", output="$out")
    read_only = ReadOnlyStep.from_step(step)

    with pytest.raises(ValidationError):
        read_only.output = "$in"


def test_run_serialization() -> None:
    """Test run can be serialized to string."""
    plan_run_id = PlanRunUUID()
    plan_run = PlanRun(
        id=plan_run_id,
        plan_id=PlanUUID(),
        end_user_id="test123",
        plan_run_inputs={"$test_input": LocalDataValue(value="input_value")},
        outputs=PlanRunOutputs(
            clarifications=[
                InputClarification(
                    plan_run_id=plan_run_id,
                    step=0,
                    argument_name="test",
                    user_guidance="help",
                    response="yes",
                    source="Test plan run",
                ),
            ],
            step_outputs={
                "1": LocalDataValue(value=ToolHardError("this is a tool hard error")),
                "2": LocalDataValue(value=ToolSoftError("this is a tool soft error")),
            },
            final_output=LocalDataValue(value="This is the end"),
        ),
    )
    assert str(plan_run) == (
        f"Run(id={plan_run.id}, plan_id={plan_run.plan_id}, "
        f"state={plan_run.state}, current_step_index={plan_run.current_step_index}, "
        f"final_output={'set' if plan_run.outputs.final_output else 'unset'})"
    )

    # check we can also serialize to JSON
    json_str = plan_run.model_dump_json()
    # parse back to run
    parsed_plan_run = PlanRun.model_validate_json(json_str)
    # ensure clarification types are maintained
    assert isinstance(parsed_plan_run.outputs.clarifications[0], InputClarification)
    # ensure plan inputs are maintained
    assert parsed_plan_run.plan_run_inputs["$test_input"].get_value() == "input_value"


def test_get_clarification_for_step_with_matching_clarification(plan_run: PlanRun) -> None:
    """Test get_clarification_for_step when there is a matching clarification."""
    # Create a clarification for step 1
    clarification = InputClarification(
        plan_run_id=plan_run.id,
        step=1,
        argument_name="test_arg",
        user_guidance="test guidance",
        resolved=False,
        source="Test plan run",
    )
    plan_run.outputs.clarifications = [clarification]

    # Get clarification for step 1
    result = plan_run.get_clarification_for_step(ClarificationCategory.INPUT)
    assert result == clarification


def test_get_clarification_for_step_without_matching_clarification(plan_run: PlanRun) -> None:
    """Test get_clarification_for_step when there is no matching clarification."""
    # Create a clarification for step 2
    clarification = InputClarification(
        plan_run_id=plan_run.id,
        step=2,
        argument_name="test_arg",
        user_guidance="test guidance",
        resolved=False,
        source="Test plan run",
    )
    plan_run.outputs.clarifications = [clarification]

    # Try to get clarification for step 1
    result = plan_run.get_clarification_for_step(ClarificationCategory.INPUT)
    assert result is None


# PlanRunV2 Tests
@pytest.fixture
def mock_end_user():
    """Create a mock EndUser for testing."""
    from portia.end_user import EndUser
    return EndUser(external_id="test_user_123", name="Test User", email="test@example.com")


@pytest.fixture
def mock_config():
    """Create a mock Config for testing."""
    from portia.config import Config
    return Config(
        llm_provider=None,
        storage_class="MEMORY",
    )


@pytest.fixture
def mock_plan_v2():
    """Create a mock PlanV2 for testing."""
    from portia.builder.plan_v2 import PlanV2
    from portia.builder.step_v2 import StepV2
    from portia.prefixed_uuid import PlanUUID

    return PlanV2(
        id=PlanUUID(),
        steps=[
            StepV2(step_name="test_step", task="Test task"),
        ],
        label="Test Plan V2"
    )


@pytest.fixture
def plan_run_v2(mock_end_user, mock_config, mock_plan_v2) -> PlanRunV2:
    """Create PlanRunV2 instance for testing."""
    return PlanRunV2(
        end_user=mock_end_user,
        config=mock_config,
        plan=mock_plan_v2,
        current_step_index=1,
        state=PlanRunState.IN_PROGRESS,
        step_output_values=[LocalDataValue(value="test_output")],
        plan_run_inputs={"$input1": LocalDataValue(value="test_input")},
    )


def test_plan_run_v2_initialization(mock_end_user, mock_config, mock_plan_v2) -> None:
    """Test initialization of PlanRunV2 instance."""
    plan_run_inputs = {"$input1": LocalDataValue(value="test_input_value")}
    plan_run_v2 = PlanRunV2(
        end_user=mock_end_user,
        config=mock_config,
        plan=mock_plan_v2,
        plan_run_inputs=plan_run_inputs,
    )

    assert plan_run_v2.id is not None
    assert plan_run_v2.plan == mock_plan_v2
    assert plan_run_v2.end_user == mock_end_user
    assert plan_run_v2.config == mock_config
    assert isinstance(plan_run_v2.plan.id.uuid, UUID)
    assert plan_run_v2.current_step_index == 0
    assert plan_run_v2.state == PlanRunState.NOT_STARTED
    assert plan_run_v2.step_output_values == []
    assert plan_run_v2.final_output is None
    assert len(plan_run_v2.plan_run_inputs) == 1
    assert plan_run_v2.plan_run_inputs["$input1"].get_value() == "test_input_value"


def test_plan_run_v2_str_representation(plan_run_v2: PlanRunV2) -> None:
    """Test string representation of PlanRunV2."""
    result = str(plan_run_v2)
    expected = (
        f"PlanRunV2(id={plan_run_v2.id}, plan_id={plan_run_v2.plan.id}, "
        f"state={plan_run_v2.state}, current_step_index={plan_run_v2.current_step_index}, "
        f"final_output=unset)"
    )
    assert result == expected


def test_plan_run_v2_str_with_final_output(plan_run_v2: PlanRunV2) -> None:
    """Test string representation of PlanRunV2 with final output set."""
    plan_run_v2.final_output = LocalDataValue(value="final result")
    result = str(plan_run_v2)
    expected = (
        f"PlanRunV2(id={plan_run_v2.id}, plan_id={plan_run_v2.plan.id}, "
        f"state={plan_run_v2.state}, current_step_index={plan_run_v2.current_step_index}, "
        f"final_output=set)"
    )
    assert result == expected


def test_plan_run_v2_str_without_plan(mock_end_user, mock_config) -> None:
    """Test string representation of PlanRunV2 without plan."""
    plan_run_v2 = PlanRunV2(
        end_user=mock_end_user,
        config=mock_config,
        plan=None,
    )
    result = str(plan_run_v2)
    expected = (
        f"PlanRunV2(id={plan_run_v2.id}, plan_id=unknown, "
        f"state={plan_run_v2.state}, current_step_index={plan_run_v2.current_step_index}, "
        f"final_output=unset)"
    )
    assert result == expected


def test_plan_run_v2_clarifications(plan_run_v2: PlanRunV2) -> None:
    """Test clarification methods in PlanRunV2."""
    # Add a clarification to the legacy outputs
    clarification = InputClarification(
        plan_run_id=plan_run_v2.id,
        step=1,
        argument_name="test_arg",
        user_guidance="test guidance",
        resolved=False,
        source="Test plan run",
    )
    plan_run_v2._legacy_outputs.clarifications = [clarification]

    # Test get_outstanding_clarifications
    outstanding = plan_run_v2.get_outstanding_clarifications()
    assert len(outstanding) == 1
    assert outstanding[0] == clarification

    # Test get_clarifications_for_step
    step_clarifications = plan_run_v2.get_clarifications_for_step(1)
    assert len(step_clarifications) == 1
    assert step_clarifications[0] == clarification

    # Test get_clarification_for_step
    result = plan_run_v2.get_clarification_for_step(ClarificationCategory.INPUT, 1)
    assert result == clarification


def test_plan_run_v2_potential_step_inputs(plan_run_v2: PlanRunV2) -> None:
    """Test get_potential_step_inputs method in PlanRunV2."""
    # Add step outputs to legacy outputs
    step_output = LocalDataValue(value="step_output_value")
    plan_run_v2._legacy_outputs.step_outputs = {"step1": step_output}

    potential_inputs = plan_run_v2.get_potential_step_inputs()

    # Should contain both step outputs and plan run inputs
    assert "step1" in potential_inputs
    assert "$input1" in potential_inputs
    assert potential_inputs["step1"] == step_output
    assert potential_inputs["$input1"] == plan_run_v2.plan_run_inputs["$input1"]


# Migration Tests
def test_migrate_plan_run_to_v2(plan_run: PlanRun, mock_end_user, mock_config, mock_plan_v2) -> None:
    """Test migration from legacy PlanRun to PlanRunV2."""
    plan_run_v2 = migrate_plan_run_to_v2(
        legacy_plan_run=plan_run,
        plan_v2=mock_plan_v2,
        end_user=mock_end_user,
        config=mock_config,
    )

    # Check all fields are migrated correctly
    assert plan_run_v2.id == plan_run.id
    assert plan_run_v2.state == plan_run.state
    assert plan_run_v2.current_step_index == plan_run.current_step_index
    assert plan_run_v2.plan == mock_plan_v2
    assert plan_run_v2.end_user == mock_end_user
    assert plan_run_v2.config == mock_config
    assert plan_run_v2.final_output == plan_run.outputs.final_output
    assert plan_run_v2.plan_run_inputs == plan_run.plan_run_inputs
    assert plan_run_v2._legacy_outputs == plan_run.outputs

    # Check step outputs are converted
    assert len(plan_run_v2.step_output_values) == len(plan_run.outputs.step_outputs)


def test_migrate_v2_to_plan_run(plan_run_v2: PlanRunV2) -> None:
    """Test migration from PlanRunV2 back to legacy PlanRun."""
    legacy_plan_run = migrate_v2_to_plan_run(plan_run_v2)

    # Check all fields are migrated correctly
    assert legacy_plan_run.id == plan_run_v2.id
    assert legacy_plan_run.plan_id == plan_run_v2.plan.id
    assert legacy_plan_run.current_step_index == plan_run_v2.current_step_index
    assert legacy_plan_run.state == plan_run_v2.state
    assert legacy_plan_run.end_user_id == plan_run_v2.end_user.external_id
    assert legacy_plan_run.outputs == plan_run_v2._legacy_outputs
    assert legacy_plan_run.plan_run_inputs == plan_run_v2.plan_run_inputs


def test_migrate_v2_to_plan_run_without_plan(mock_end_user, mock_config) -> None:
    """Test migration from PlanRunV2 to legacy PlanRun when plan is None."""
    plan_run_v2 = PlanRunV2(
        end_user=mock_end_user,
        config=mock_config,
        plan=None,
    )

    legacy_plan_run = migrate_v2_to_plan_run(plan_run_v2)

    assert legacy_plan_run.plan_id == "unknown"
    assert legacy_plan_run.end_user_id == mock_end_user.external_id


def test_migration_roundtrip(plan_run: PlanRun, mock_end_user, mock_config, mock_plan_v2) -> None:
    """Test that migration works correctly in both directions."""
    # Migrate to V2 and back
    plan_run_v2 = migrate_plan_run_to_v2(
        legacy_plan_run=plan_run,
        plan_v2=mock_plan_v2,
        end_user=mock_end_user,
        config=mock_config,
    )
    migrated_back = migrate_v2_to_plan_run(plan_run_v2)

    # Check that key fields are preserved after roundtrip
    assert migrated_back.id == plan_run.id
    assert migrated_back.current_step_index == plan_run.current_step_index
    assert migrated_back.state == plan_run.state
    assert migrated_back.outputs.clarifications == plan_run.outputs.clarifications
    assert migrated_back.outputs.step_outputs == plan_run.outputs.step_outputs
    assert migrated_back.plan_run_inputs == plan_run.plan_run_inputs
