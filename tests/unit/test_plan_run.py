"""Tests for Run primitives."""

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from portia.clarification import Clarification, ClarificationCategory, InputClarification
from portia.errors import ToolHardError, ToolSoftError
from portia.execution_agents.output import LocalDataValue, Output
from portia.plan import PlanUUID, ReadOnlyStep, Step
from portia.plan_run import PlanRun, PlanRunOutputs, PlanRunState, ReadOnlyPlanRun
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
            step_outputs={
                "step1": Output(name="step1", step=0, value=LocalDataValue(value="Test output"))
            },
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
                "1": Output(
                    name="1",
                    step=0,
                    value=LocalDataValue(value=ToolHardError("this is a tool hard error")),
                ),
                "2": Output(
                    name="2",
                    step=1,
                    value=LocalDataValue(value=ToolSoftError("this is a tool soft error")),
                ),
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


def test_plan_run_outputs_for_step_found() -> None:
    """Test for_step returns the correct Output when the step exists."""
    output0 = Output(name="step0", step=0, value=LocalDataValue(value="output0"))
    output1 = Output(name="step1", step=1, value=LocalDataValue(value="output1"))
    outputs = PlanRunOutputs(
        step_outputs={
            "step0": output0,
            "step1": output1,
        }
    )
    result = outputs.for_step(1)
    assert result == output1


def test_plan_run_outputs_for_step_not_found() -> None:
    """Test for_step returns None when the step does not exist."""
    output0 = Output(name="step0", step=0, value=LocalDataValue(value="output0"))
    outputs = PlanRunOutputs(
        step_outputs={
            "step0": output0,
        }
    )
    result = outputs.for_step(2)
    assert result is None
