"""Workflows are executing instances of a Plan.

A workflow encapsulates all execution state, serving as the definitive record of its progress.
As the workflow runs, its `WorkflowState`, `current_step_index`, and `outputs` evolve to reflect
the current execution state.

The workflow also retains an `ExecutionContext`, which provides valuable insights for debugging
and analytics, capturing contextual information relevant to the workflow's execution.

Key Components
--------------
- **WorkflowState**: Tracks the current status of the workflow (e.g., NOT_STARTED, IN_PROGRESS).
- **current_step_index**: Represents the step within the plan currently being executed.
- **outputs**: Stores the intermediate and final results of the workflow.
- **ExecutionContext**: Provides contextual metadata useful for logging and performance analysis.
"""

from __future__ import annotations

import json
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field

from portia.agents.base_agent import Output
from portia.clarification import (
    ClarificationListType,
)
from portia.common import PortiaEnum, PrefixedUUID
from portia.execution_context import ExecutionContext, empty_context
from portia.plan import PlanUUID

WORKFLOW_UUID_PREFIX = "wkfl"


class WorkflowState(PortiaEnum):
    """The current state of the Workflow.

    Attributes
    ----------
    NOT_STARTED : str
        The workflow has not been started yet.
    IN_PROGRESS : str
        The workflow is currently in progress.
    NEED_CLARIFICATION : str
        The workflow requires further clarification before proceeding.
    READY_TO_RESUME : str
        The workflow is ready to resume after clarifications have been resolved.
    COMPLETE : str
        The workflow has been successfully completed.
    FAILED : str
        The workflow has encountered an error and failed.

    """

    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    NEED_CLARIFICATION = "NEED_CLARIFICATION"
    READY_TO_RESUME = "READY_TO_RESUME"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class WorkflowOutputs(BaseModel):
    """Outputs of a workflow, including clarifications.

    Attributes
    ----------
    clarifications : ClarificationListType
        Clarifications raise by this workflow.
    step_outputs : dict[str, Output]
        A dictionary containing outputs of individual workflow steps.
        Outputs are indexed by the value given by the `step.output` field of the plan.
    final_output : Output | None
        The final consolidated output of the workflow, if available.

    """

    model_config = ConfigDict(extra="forbid")

    clarifications: ClarificationListType = Field(
        default=[],
        description="Clarifications raise by this workflow.",
    )

    step_outputs: dict[str, Output] = Field(
        default={},
        description="A dictionary containing outputs of individual workflow steps.",
    )

    final_output: Output | None = Field(
        default=None,
        description="The final consolidated output of the workflow, if available.",
    )


class Workflow(BaseModel):
    """A workflow represents a running instance of a Plan.

    Attributes
    ----------
    id : WorkflowUUID
        A unique ID for this workflow.
    plan_id : PlanUUID
        The ID of the Plan this Workflow uses.
    current_step_index : int
        The current step that is being executed.
    state : WorkflowState
        The current state of the workflow.
    execution_context : ExecutionContext
        Execution context for the workflow.
    outputs : WorkflowOutputs
        Outputs of the workflow, including clarifications.

    """

    model_config = ConfigDict(extra="forbid")

    id: WorkflowUUID = Field(
        default_factory=lambda: WorkflowUUID(),
        description="A unique ID for this workflow.",
    )
    plan_id: PlanUUID = Field(
        description="The ID of the Plan this Workflow uses.",
    )
    current_step_index: int = Field(
        default=0,
        description="The current step that is being executed",
    )
    state: WorkflowState = Field(
        default=WorkflowState.NOT_STARTED,
        description="The current state of the workflow.",
    )
    execution_context: ExecutionContext = Field(
        default=empty_context(),
        description="Execution Context for the workflow.",
    )
    outputs: WorkflowOutputs = Field(
        default=WorkflowOutputs(),
        description="Outputs of the workflow including clarifications.",
    )

    def get_outstanding_clarifications(self) -> ClarificationListType:
        """Return all outstanding clarifications.

        Returns
        -------
        ClarificationListType
            A list of outstanding clarifications that have not been resolved.

        """
        return [
            clarification
            for clarification in self.outputs.clarifications
            if not clarification.resolved
        ]

    '''
    @classmethod
    def model_validate_json(cls: type[Self], json_data: str | bytes) -> Self:
        """Validate JSON and deserialize the UUID field."""
        if isinstance(json_data, bytes):
            json_data = json_data.decode()
        data = json.loads(json_data)
        if isinstance(data.get("id"), str):
            data["id"] = WorkflowUUID.from_string(data["id"])
        if isinstance(data.get("plan_id"), str):
            data["plan_id"] = PlanUUID.from_string(data["plan_id"])
        return cls.model_validate(data)
    '''

    def __str__(self) -> str:
        """Return the string representation of the workflow.

        Returns
        -------
        str
            A string representation containing key workflow attributes.

        """
        return (
            f"Workflow(id={self.id}, plan_id={self.plan_id}, "
            f"state={self.state}, current_step_index={self.current_step_index}, "
            f"final_output={'set' if self.outputs.final_output else 'unset'})"
        )


class ReadOnlyWorkflow(Workflow):
    """A read-only copy of a workflow, passed to agents for reference.

    This class provides a non-modifiable view of a workflow instance,
    ensuring that agents can access workflow details without altering them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def from_workflow(cls, workflow: Workflow) -> ReadOnlyWorkflow:
        """Create a read-only workflow from a normal workflow.

        Parameters
        ----------
        workflow : Workflow
            The original workflow instance to create a read-only copy from.

        Returns
        -------
        ReadOnlyWorkflow
            A new read-only instance of the provided workflow.

        """
        return cls(
            id=workflow.id,
            plan_id=workflow.plan_id,
            current_step_index=workflow.current_step_index,
            outputs=workflow.outputs,
            state=workflow.state,
            execution_context=workflow.execution_context,
        )

class WorkflowUUID(PrefixedUUID):
    """A UUID for a workflow."""

    prefix: ClassVar[str] = WORKFLOW_UUID_PREFIX
