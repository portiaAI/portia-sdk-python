"""Tool Wrapper that intecepts run calls and records them."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict

from portia.clarification import Clarification
from portia.common import combine_args_kwargs
from portia.storage import ToolCallRecord, ToolCallState, ToolCallStorage
from portia.tool import Tool

if TYPE_CHECKING:
    from portia.context import ExecutionContext
    from portia.workflow import Workflow


class ToolCallWrapper(Tool):
    """Tool Wrapper that records calls to its child tool and sends them to the ToolCallStorage."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # set these as private to allow pydantic init
    _child_tool: Tool
    _storage: ToolCallStorage
    _workflow: Workflow

    def __init__(self, child_tool: Tool, storage: ToolCallStorage, workflow: Workflow) -> None:
        """Initialize parent fields using child_tool's attributes."""
        super().__init__(
            id=child_tool.id,
            name=child_tool.name,
            description=child_tool.description,
            args_schema=child_tool.args_schema,
            output_schema=child_tool.output_schema,
        )
        self._child_tool = child_tool
        self._storage = storage
        self._workflow = workflow

    def run(self, ctx: ExecutionContext, *args: Any, **kwargs: Any) -> Any | Clarification:  # noqa: ANN401
        """Run the child tool and store the outcome."""

        # initialize empty call record
        record = ToolCallRecord(
            input=combine_args_kwargs(*args, **kwargs),
            output=None,
            latency=0,
            tool_name=self._child_tool.name,
            workflow_id=self._workflow.id,
            step=self._workflow.current_step_index,
            end_user_id=ctx.end_user_id,
            additional_data=ctx.additional_data,
            outcome=ToolCallState.IN_PROGRESS,
        )
        start_time = datetime.now(tz=UTC)
        try:
            output = self._child_tool.run(ctx, *args, **kwargs)
        except Exception as e:
            record.output = e
            record.latency = (datetime.now(tz=UTC) - start_time).microseconds
            record.outcome = ToolCallState.FAILED
            self._storage.save_tool_call(record)
            raise
        else:
            record.output = output
            if isinstance(output, Clarification):
                record.outcome = ToolCallState.NEED_CLARIFICATION
            else:
                record.outcome = ToolCallState.SUCCESS
            record.latency = (datetime.now(tz=UTC) - start_time).microseconds
            self._storage.save_tool_call(record)
        return output
