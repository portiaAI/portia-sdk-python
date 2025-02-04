"""Tool Call module contains classes that record the outcome of a single tool call.

The `ToolCallStatus` enum defines the various states a tool call can be in, such
as in progress, successful, requiring clarification, or failing.

The `ToolCallRecord` class is a Pydantic model used to capture details about a
specific tool call, including its status, input, output, and associated metadata.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict

from portia.common import PortiaEnum
from portia.workflow import WorkflowUUID


class ToolCallStatus(PortiaEnum):
    """The status of the tool call.

    Attributes
    ----------
    IN_PROGRESS : str
        The tool is currently in progress.
    NEED_CLARIFICATION : str
        The tool raise a clarification.
    SUCCESS : str
        The tool executed successfully.
    FAILED : str
        The tool raised an error.

    """

    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    NEED_CLARIFICATION = "NEED_CLARIFICATION"
    FAILED = "FAILED"


class ToolCallRecord(BaseModel):
    """Model that records the details of an individual tool call.

    This class captures all relevant information about a single tool call
    within a workflow, including metadata, input and output data, and status.

    Attributes
    ----------
        tool_name (str): The name of the tool being called.
        workflow_id (WorkflowUUID): The unique identifier of the workflow to which this tool call
            belongs.
        step (int): The step number of the tool call in the workflow.
        end_user_id (str | None): The ID of the end user, if applicable. Can be None.
        additional_data (dict[str, str]): Additional data from the execution context.
        status (ToolCallStatus): The current status of the tool call (e.g., IN_PROGRESS, SUCCESS).
        input (Any): The input data passed to the tool call.
        output (Any): The output data returned from the tool call.
        latency_seconds (float): The latency in seconds for the tool call to complete.

    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    workflow_id: WorkflowUUID
    step: int
    # execution context is tracked here so we get a snapshot if its updated
    end_user_id: str | None
    additional_data: dict[str, str]
    # details of the tool call are below
    status: ToolCallStatus
    input: Any
    output: Any
    latency_seconds: float
