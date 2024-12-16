"""Workflow primitives."""

from __future__ import annotations

from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from portia.clarification import Clarification
from portia.plan import Output


class WorkflowState(str, Enum):
    """Progress of the Workflow."""

    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    NEED_CLARIFICATION = "NEED_CLARIFICATION"
    FAILED = "FAILED"


class InvalidWorkflowStateError(Exception):
    """The given workflow is in an invalid state."""


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
    step_outputs: dict[str, Output] = {}
