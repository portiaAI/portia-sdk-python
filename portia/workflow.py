"""Workflow primitives."""

from __future__ import annotations

from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from portia.agents.base_agent import Output
from portia.clarification import ActionClarification, InputClarification, MultiChoiceClarification, Clarification
from portia.context import ExecutionContext, empty_context


class WorkflowState(str, Enum):
    """Progress of the Workflow."""

    @classmethod
    def enumerate(cls) -> tuple[tuple[str, str], ...]:
        """Return a tuple of all choices as (name, value) pairs."""
        return tuple((x.name, x.value) for x in cls)

    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    NEED_CLARIFICATION = "NEED_CLARIFICATION"
    FAILED = "FAILED"


class WorkflowOutputs(BaseModel):
    """Outputs of a workflow including clarifications."""

    model_config = ConfigDict(extra="forbid")

    clarifications: list[ActionClarification | InputClarification | MultiChoiceClarification | Clarification] = Field(
        default=[],
        description="Any clarifications needed for this workflow.",
    )

    step_outputs: dict[str, Output] = {}

    final_output: Output | None = None

    def to_json(self) -> dict:
        """Convert the outputs to a JSON serializable dictionary."""
        return self.model_dump(mode="json")


class Workflow(BaseModel):
    """A workflow represent a running instance of a Plan."""

    id: UUID = Field(
        default_factory=uuid4,
        description="A unique ID for this workflow.",
    )
    plan_id: UUID = Field(
        description="The plan this relates to",
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

    def get_outstanding_clarifications(self) -> list[Clarification]:
        """Return all outstanding clarifications."""
        return [
            clarification
            for clarification in self.outputs.clarifications
            if not clarification.resolved
        ]


class ReadOnlyWorkflow(Workflow):
    """A read only copy of a workflow, passed to agents for reference."""

    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_workflow(cls, workflow: Workflow) -> ReadOnlyWorkflow:
        """Configure a read only workflow from a normal workflow."""
        return cls(
            id=workflow.id,
            plan_id=workflow.plan_id,
            current_step_index=workflow.current_step_index,
            outputs=workflow.outputs,
            state=workflow.state,
            execution_context=workflow.execution_context,
        )
