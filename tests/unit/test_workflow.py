"""Tests for Workflow primitives."""

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from portia.agents.base_agent import Output
from portia.clarification import Clarification, InputClarification
from portia.errors import ToolHardError, ToolSoftError
from portia.plan import PlanUUID, ReadOnlyStep, Step
from portia.workflow import ReadOnlyWorkflow, Workflow, WorkflowOutputs, WorkflowState


@pytest.fixture
def mock_clarification() -> InputClarification:
    """Create a mock clarification for testing."""
    return InputClarification(user_guidance="test", resolved=False, argument_name="test")


@pytest.fixture
def workflow(mock_clarification: InputClarification) -> Workflow:
    """Create a Workflow instance for testing."""
    return Workflow(
        plan_id=PlanUUID(),
        current_step_index=1,
        state=WorkflowState.IN_PROGRESS,
        outputs=WorkflowOutputs(
            clarifications=[mock_clarification],
            step_outputs={"step1": Output(value="Test output")},
        ),
    )


def test_workflow_initialization() -> None:
    """Test initialization of a Workflow instance."""
    plan_id = PlanUUID()
    workflow = Workflow(plan_id=plan_id)

    assert workflow.id is not None
    assert workflow.plan_id == plan_id
    assert isinstance(workflow.plan_id.uuid, UUID)
    assert workflow.current_step_index == 0
    assert workflow.outputs.clarifications == []
    assert workflow.state == WorkflowState.NOT_STARTED
    assert workflow.outputs.step_outputs == {}


def test_workflow_get_outstanding_clarifications(
    workflow: Workflow,
    mock_clarification: Clarification,
) -> None:
    """Test get_outstanding_clarifications method."""
    outstanding = workflow.get_outstanding_clarifications()

    assert len(outstanding) == 1
    assert outstanding[0] == mock_clarification


def test_workflow_get_outstanding_clarifications_none() -> None:
    """Test get_outstanding_clarifications when no clarifications are outstanding."""
    workflow = Workflow(plan_id=PlanUUID(), outputs=WorkflowOutputs(clarifications=[]))

    assert workflow.get_outstanding_clarifications() == []


def test_workflow_state_enum() -> None:
    """Test the WorkflowState enum values."""
    assert WorkflowState.NOT_STARTED == "NOT_STARTED"
    assert WorkflowState.IN_PROGRESS == "IN_PROGRESS"
    assert WorkflowState.COMPLETE == "COMPLETE"
    assert WorkflowState.NEED_CLARIFICATION == "NEED_CLARIFICATION"
    assert WorkflowState.FAILED == "FAILED"


def test_read_only_workflow_immutable() -> None:
    """Test immutability of workflow."""
    workflow = Workflow(plan_id=PlanUUID(uuid=uuid4()))
    read_only = ReadOnlyWorkflow.from_workflow(workflow)

    with pytest.raises(ValidationError):
        read_only.state = WorkflowState.IN_PROGRESS


def test_read_only_step_immutable() -> None:
    """Test immutability of step."""
    step = Step(task="add", output="$out")
    read_only = ReadOnlyStep.from_step(step)

    with pytest.raises(ValidationError):
        read_only.output = "$in"


def test_workflow_serialization() -> None:
    """Test workflow can be serialized to string."""
    workflow = Workflow(
        plan_id=PlanUUID(),
        outputs=WorkflowOutputs(
            clarifications=[
                InputClarification(
                    step=0,
                    argument_name="test",
                    user_guidance="help",
                    response="yes",
                ),
            ],
            step_outputs={
                "1": Output(value=ToolHardError("this is a tool hard error")),
                "2": Output(value=ToolSoftError("this is a tool soft error")),
            },
            final_output=Output(value="This is the end"),
        ),
    )
    assert str(workflow) == (
        f"Workflow(id={workflow.id}, plan_id={workflow.plan_id}, "
        f"state={workflow.state}, current_step_index={workflow.current_step_index}, "
        f"final_output={'set' if workflow.outputs.final_output else 'unset'})"
    )
    # check we can also serialize to JSON
    json_str = workflow.model_dump_json()
    # parse back to workflow
    parsed_workflow = Workflow.model_validate_json(json_str)
    # ensure clarification types are maintained
    assert isinstance(parsed_workflow.outputs.clarifications[0], InputClarification)
