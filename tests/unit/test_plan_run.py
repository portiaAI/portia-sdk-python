"""Tests for Run primitives."""

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from portia.clarification import Clarification, InputClarification
from portia.errors import ToolHardError, ToolSoftError
from portia.execution_agents.output import LocalOutput
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
    )


@pytest.fixture
def plan_run(mock_clarification: InputClarification) -> PlanRun:
    """Create PlanRun instance for testing."""
    return PlanRun(
        plan_id=PlanUUID(),
        current_step_index=1,
        state=PlanRunState.IN_PROGRESS,
        outputs=PlanRunOutputs(
            clarifications=[mock_clarification],
            step_outputs={"step1": LocalOutput(value="Test output")},
        ),
    )


def test_run_initialization() -> None:
    """Test initialization of PlanRun instance."""
    plan_id = PlanUUID()
    plan_run = PlanRun(plan_id=plan_id)

    assert plan_run.id is not None
    assert plan_run.plan_id == plan_id
    assert isinstance(plan_run.plan_id.uuid, UUID)
    assert plan_run.current_step_index == 0
    assert plan_run.outputs.clarifications == []
    assert plan_run.state == PlanRunState.NOT_STARTED
    assert plan_run.outputs.step_outputs == {}


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
    plan_run = PlanRun(plan_id=PlanUUID(), outputs=PlanRunOutputs(clarifications=[]))

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
    plan_run = PlanRun(plan_id=PlanUUID(uuid=uuid4()))
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
        outputs=PlanRunOutputs(
            clarifications=[
                InputClarification(
                    plan_run_id=plan_run_id,
                    step=0,
                    argument_name="test",
                    user_guidance="help",
                    response="yes",
                ),
            ],
            step_outputs={
                "1": LocalOutput(value=ToolHardError("this is a tool hard error")),
                "2": LocalOutput(value=ToolSoftError("this is a tool soft error")),
            },
            final_output=LocalOutput(value="This is the end"),
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
