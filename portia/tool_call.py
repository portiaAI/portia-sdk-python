"""Tool Calls record the outcome of a single tool call."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel

from portia.common import PortiaEnum


class ToolCallStatus(PortiaEnum):
    """State of a tool call."""

    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    NEED_CLARIFICATION = "NEED_CLARIFICATION"
    FAILED = "FAILED"


class ToolCallRecord(BaseModel):
    """Records an individual tool call."""

    tool_name: str
    workflow_id: UUID
    step: int
    # execution context is tracked here so we get a snapshot if its updated
    end_user_id: str | None
    additional_data: dict[str, str]
    # details of the tool call are below
    status: ToolCallStatus
    input: Any
    output: Any
    latency_seconds: float
