"""Workflow primitives."""

from __future__ import annotations

from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from portia.agents.base_agent import Output
from portia.clarification import Clarification
from portia.context import ExecutionContext, empty_context


class WorkflowState(str, Enum):
    """Progress of the Workflow."""

    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    NEED_CLARIFICATION = "NEED_CLARIFICATION"
    FAILED = "FAILED"


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
    clarifications: list[Clarification] = Field(
        default=[],
        description="Any clarifications needed for this workflow.",
    )
    state: WorkflowState = Field(
        default=WorkflowState.NOT_STARTED,
        description="The current state of the workflow.",
    )
    execution_context: ExecutionContext = Field(
        default=empty_context(),
        description="Execution Context for the workflow.",
    )

    step_outputs: dict[str, Output] = {}

    final_output: Output | None = None

    def get_outstanding_clarifications(self) -> list[Clarification]:
        """Return all outstanding clarifications."""
        return [
            clarification for clarification in self.clarifications if not clarification.resolved
        ]

    def __str__(self) -> str:
        """Return the string representation."""
        return (
            f"Workflow(id={self.id}, plan_id={self.plan_id}, "
            f"state={self.state}, current_step_index={self.current_step_index}, "
            f"final_output={'set' if self.final_output else 'unset'})"
        )


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
            clarifications=workflow.clarifications,
            state=workflow.state,
            step_outputs=workflow.step_outputs,
            final_output=workflow.final_output,
        )
