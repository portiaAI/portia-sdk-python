"""Tool Call module contains classes that record the outcome of a single tool call.

The `ToolCallStatus` enum defines the various states a tool call can be in, such
as in progress, successful, requiring clarification, or failing.

The `ToolCallRecord` class is a Pydantic model used to capture details about a
specific tool call, including its status, input, output, and associated metadata.
"""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from portia.common import PortiaEnum
from portia.plan_run import PlanRunUUID


class ToolCallStatus(PortiaEnum):
    """The status of the tool call.

    Attributes:
        IN_PROGRESS: The tool is currently in progress.
        NEED_CLARIFICATION: The tool raise a clarification.
        SUCCESS: The tool executed successfully.
        FAILED: The tool raised an error.

    """

    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    NEED_CLARIFICATION = "NEED_CLARIFICATION"
    FAILED = "FAILED"


class ToolCallRecord(BaseModel):
    """Model that records the details of an individual tool call.

    This class captures all relevant information about a single tool call
    within a PlanRun including metadata, input and output data, and status.

    Attributes:
        tool_name (str): The name of the tool being called.
        plan_run_id (RunUUID): The unique identifier of the run to which this tool call
            belongs.
        step (int): The step number of the tool call in the PlanRun.
        end_user_id (str | None): The ID of the end user, if applicable. Can be None.
        status (ToolCallStatus): The current status of the tool call (e.g., IN_PROGRESS, SUCCESS).
        input (Any): The input data passed to the tool call.
        output (Any): The output data returned from the tool call.
        latency_seconds (float): The latency in seconds for the tool call to complete.

    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    plan_run_id: PlanRunUUID
    step: int
    end_user_id: str | None
    # details of the tool call are below
    status: ToolCallStatus
    input: Any
    output: Any
    latency_seconds: float

    def serialize_input(self) -> Any:  # noqa: ANN401
        """Handle serialization of inputs."""
        return self._serialize_value(self.input)

    def serialize_output(self) -> Any:  # noqa: ANN401
        """Handle serialization of outputs."""
        return self._serialize_value(self.output)

    def _serialize_value(self, value: Any) -> Any:  # noqa: ANN401
        """Handle serialization of inputs/outputs."""
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")

        # If we can JSON dumps here it means we can just return the
        # raw value
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            return f"<<UNSERIALIZABLE: {type(value).__name__}>>"
        else:
            return value
