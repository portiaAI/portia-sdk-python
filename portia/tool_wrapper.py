"""Tool Wrapper that intercepts run calls and records them.

This module contains the `ToolCallWrapper` class, which wraps around an existing tool and records
information about the tool's execution, such as input, output, latency, and status. The recorded
data is stored in `AdditionalStorage` for later use.

Classes:
    ToolCallWrapper: A wrapper that intercepts tool calls, records execution data, and stores it.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict

from portia.common import combine_args_kwargs
from portia.execution_agents.execution_utils import is_clarification
from portia.logger import logger
from portia.storage import AdditionalStorage, ToolCallRecord, ToolCallStatus
from portia.tool import ReadyResponse, Tool, ToolRunContext

if TYPE_CHECKING:
    from portia.clarification import Clarification
    from portia.execution_agents.output import Output
    from portia.plan_run import PlanRun


MAX_OUTPUT_LOG_LENGTH = 1000


class ToolCallWrapper(Tool):
    """Tool Wrapper that records calls to its child tool and sends them to the AdditionalStorage.

    This class is a wrapper around a child tool. It captures the input and output, measures latency,
    and records the status of the execution. The results are then stored in the provided
    `AdditionalStorage`.

    Attributes:
        model_config (ConfigDict): Pydantic configuration that allows arbitrary types.
        _child_tool (Tool): The child tool to be wrapped and executed.
        _storage (AdditionalStorage): Storage mechanism to save tool call records.
        _plan_run (PlanRun): The run context for the current execution.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _child_tool: Tool
    _storage: AdditionalStorage
    _plan_run: PlanRun

    def __init__(self, child_tool: Tool, storage: AdditionalStorage, plan_run: PlanRun) -> None:
        """Initialize parent fields using child_tool's attributes.

        Args:
            child_tool (Tool): The child tool to be wrapped.
            storage (AdditionalStorage): The storage to save execution records.
            plan_run (PlanRun): The PlanRun to execute.

        """
        super().__init__(
            id=child_tool.id,
            name=child_tool.name,
            description=child_tool.description,
            args_schema=child_tool.args_schema,
            output_schema=child_tool.output_schema,
            should_summarize=child_tool.should_summarize,
            structured_output_schema=child_tool.structured_output_schema,
        )
        self._child_tool = child_tool
        self._storage = storage
        self._plan_run = plan_run

    def ready(self, ctx: ToolRunContext) -> ReadyResponse:
        """Check if the child tool is ready and return ReadyResponse."""
        return self._child_tool.ready(ctx)

    def run(self, ctx: ToolRunContext, *args: Any, **kwargs: Any) -> Any | Clarification:  # noqa: ANN401
        """Run the child tool and store the outcome.

        This method executes the child tool with the provided arguments, records the input,
        output, latency, and status of the execution, and stores the details in `AdditionalStorage`.

        Args:
            ctx (ToolRunContext): The context containing user data and metadata.
            *args (Any): Positional arguments for the child tool.
            **kwargs (Any): Keyword arguments for the child tool.

        Returns:
            Any | Clarification: The output of the child tool or a clarification request.

        Raises:
            Exception: If an error occurs during execution, the exception is logged, and the
                status is set to `FAILED`.

        """
        # initialize empty call record
        record = self._setup_tool_call(ctx, *args, **kwargs)
        start_time = datetime.now(tz=UTC)
        try:
            output = self._child_tool._run(ctx, *args, **kwargs)  # noqa: SLF001
        except Exception as e:
            # handle the tool call record
            record.output = str(e)
            record.latency_seconds = (datetime.now(tz=UTC) - start_time).total_seconds()
            record.status = ToolCallStatus.FAILED
            raise
        else:
            record = self._process_output(record, output, start_time)
        finally:
            self._storage.save_tool_call(record)
            self._storage.save_end_user(ctx.end_user)

        return output.get_value()

    def _process_output(
        self,
        record: ToolCallRecord,
        output: Output,
        start_time: datetime,
    ) -> ToolCallRecord:
        """Process the output of the tool call and update the record."""
        output_value = output.get_value()
        if is_clarification(output_value) and output_value is not None:
            record.output = [clar.model_dump(mode="json") for clar in output_value]
            record.status = ToolCallStatus.NEED_CLARIFICATION
        elif output_value is None:
            record.output = output.model_dump(mode="json")
            record.status = ToolCallStatus.SUCCESS
        else:
            record.output = output_value
            record.status = ToolCallStatus.SUCCESS
        record.latency_seconds = (datetime.now(tz=UTC) - start_time).total_seconds()
        return record

    def _setup_tool_call(self, ctx: ToolRunContext, *args: Any, **kwargs: Any) -> ToolCallRecord:
        record = ToolCallRecord(
            input=combine_args_kwargs(*args, **kwargs),
            output=None,
            latency_seconds=0,
            tool_name=self._child_tool.name,
            plan_run_id=self._plan_run.id,
            step=self._plan_run.current_step_index,
            end_user_id=ctx.end_user.external_id,
            status=ToolCallStatus.IN_PROGRESS,
        )
        input_str = repr(record.input)
        truncated_input = (
            input_str[:MAX_OUTPUT_LOG_LENGTH] + "...[truncated - only first 1000 characters shown]"
            if len(input_str) > MAX_OUTPUT_LOG_LENGTH
            else input_str
        )
        logger().info(
            f"Invoking {record.tool_name!s} with args: {truncated_input}",
        )
        return record

    async def arun(self, ctx: ToolRunContext, *args: Any, **kwargs: Any) -> Any | Clarification:  # noqa: ANN401
        """Async run the child tool and store the outcome."""
        record = self._setup_tool_call(ctx, *args, **kwargs)
        start_time = datetime.now(tz=UTC)
        try:
            output = await self._child_tool._arun(ctx, *args, **kwargs)  # noqa: SLF001
        except Exception as e:
            record.output = str(e)
            record.latency_seconds = (datetime.now(tz=UTC) - start_time).total_seconds()
            record.status = ToolCallStatus.FAILED
            raise
        else:
            record = self._process_output(record, output, start_time)
        finally:
            await asyncio.gather(
                self._storage.asave_tool_call(record),
                self._storage.asave_end_user(ctx.end_user),
            )

        return output.get_value()
