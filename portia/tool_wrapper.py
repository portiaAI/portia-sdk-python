"""Tool Wrapper that intercepts run calls and records them.

This module contains the `ToolCallWrapper` class, which wraps around an existing tool and records
information about the tool's execution, such as input, output, latency, and status. The recorded
data is stored in `ToolCallStorage` for later use.

Classes:
    ToolCallWrapper: A wrapper that intercepts tool calls, records execution data, and stores it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict

from portia.clarification import Clarification
from portia.common import combine_args_kwargs
from portia.execution_context import ExecutionContext
from portia.storage import ToolCallRecord, ToolCallStatus, ToolCallStorage
from portia.tool import Tool

if TYPE_CHECKING:
    from portia.execution_context import ExecutionContext
    from portia.workflow import Workflow


class ToolCallWrapper(Tool):
    """Tool Wrapper that records calls to its child tool and sends them to the ToolCallStorage.

    This class is a wrapper around a child tool. It captures the input and output, measures latency,
    and records the status of the execution. The results are then stored in the provided
    `ToolCallStorage`.

    Attributes
    ----------
        model_config (ConfigDict): Pydantic configuration that allows arbitrary types.
        _child_tool (Tool): The child tool to be wrapped and executed.
        _storage (ToolCallStorage): Storage mechanism to save tool call records.
        _workflow (Workflow): The workflow context for the current execution.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _child_tool: Tool
    _storage: ToolCallStorage
    _workflow: Workflow

    def __init__(self, child_tool: Tool, storage: ToolCallStorage, workflow: Workflow) -> None:
        """Initialize parent fields using child_tool's attributes.

        Args:
            child_tool (Tool): The child tool to be wrapped.
            storage (ToolCallStorage): The storage to save execution records.
            workflow (Workflow): The current workflow for the execution.

        """
        super().__init__(
            id=child_tool.id,
            name=child_tool.name,
            description=child_tool.description,
            args_schema=child_tool.args_schema,
            output_schema=child_tool.output_schema,
            should_summarize=child_tool.should_summarize,
        )
        self._child_tool = child_tool
        self._storage = storage
        self._workflow = workflow

    def ready(self, ctx: ExecutionContext) -> bool:
        """Check if the child tool is ready.

        Args:
            ctx (ExecutionContext): Context of the execution environment

        Returns:
            bool: Whether the tool is ready to run

        """
        return self._child_tool.ready(ctx)

    def run(self, ctx: ExecutionContext, *args: Any, **kwargs: Any) -> Any | Clarification:  # noqa: ANN401
        """Run the child tool and store the outcome.

        This method executes the child tool with the provided arguments, records the input,
        output, latency, and status of the execution, and stores the details in `ToolCallStorage`.

        Args:
            ctx (ExecutionContext): The execution context containing user data and metadata.
            *args (Any): Positional arguments for the child tool.
            **kwargs (Any): Keyword arguments for the child tool.

        Returns:
            Any | Clarification: The output of the child tool or a clarification request.

        Raises:
            Exception: If an error occurs during execution, the exception is logged, and the
                status is set to `FAILED`.

        """
        # initialize empty call record
        record = ToolCallRecord(
            input=combine_args_kwargs(*args, **kwargs),
            output=None,
            latency_seconds=0,
            tool_name=self._child_tool.name,
            workflow_id=self._workflow.id,
            step=self._workflow.current_step_index,
            end_user_id=ctx.end_user_id,
            additional_data=ctx.additional_data,
            status=ToolCallStatus.IN_PROGRESS,
        )
        start_time = datetime.now(tz=UTC)
        try:
            output = self._child_tool.run(ctx, *args, **kwargs)
        except Exception as e:
            record.output = str(e)
            record.latency_seconds = (datetime.now(tz=UTC) - start_time).total_seconds()
            record.status = ToolCallStatus.FAILED
            self._storage.save_tool_call(record)
            raise
        else:
            if isinstance(output, Clarification):
                record.status = ToolCallStatus.NEED_CLARIFICATION
                record.output = output.model_dump(mode="json")
            else:
                record.output = output
                record.status = ToolCallStatus.SUCCESS
            record.latency_seconds = (datetime.now(tz=UTC) - start_time).total_seconds()
            self._storage.save_tool_call(record)
        return output
