"""Implementation of the invoke tool step."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from portia.builder.step import Step

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override  # pragma: no cover

from langsmith import traceable
from pydantic import BaseModel, Field

from portia.clarification import (
    Clarification,
)
from portia.errors import ToolNotFoundError
from portia.execution_agents.execution_utils import is_clarification
from portia.model import Message
from portia.plan import Step as StepData
from portia.tool import Tool  # noqa: TC001
from portia.tool_wrapper import ToolCallWrapper

if TYPE_CHECKING:
    from portia.builder.plan import Plan
    from portia.run_context import RunContext


class InvokeToolStep(Step):
    """A step that directly invokes a tool with specific arguments.

    This performs a direct tool call without LLM involvement, making it suitable
    for deterministic operations where you know exactly which tool to call and
    what arguments to pass.
    """

    tool: str | Tool = Field(
        description=(
            "The tool to invoke. Can be either a tool ID string (to lookup in the tool registry) "
            "or a Tool instance to run directly."
        )
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arguments to pass to the tool. Values can be references to previous step outputs "
            "(using StepOutput), plan inputs (using Input), or literal values."
        ),
    )
    output_schema: type[BaseModel] | None = Field(
        default=None,
        description=(
            "Pydantic model class to structure the tool's output. If provided, the raw tool "
            "output will be converted to match this schema."
        ),
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        tool_name = self.tool if isinstance(self.tool, str) else self.tool.id
        return f"InvokeToolStep(tool='{tool_name}', args={self.args}{output_info})"

    @override
    @traceable(name="Invoke Tool Step - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Execute the tool and return its result."""
        if isinstance(self.tool, str):
            tool = ToolCallWrapper.from_tool_id(
                self.tool,
                run_data.tool_registry,
                run_data.storage,
                run_data.plan_run,
            )
        else:
            tool = ToolCallWrapper(
                child_tool=self.tool,
                storage=run_data.storage,
                plan_run=run_data.plan_run,
            )
        if not tool:
            raise ToolNotFoundError(self.tool if isinstance(self.tool, str) else self.tool.id)

        tool_ctx = run_data.get_tool_run_ctx()
        args = {k: self._resolve_references(v, run_data) for k, v in self.args.items()}
        output = await tool._arun(tool_ctx, **args)  # noqa: SLF001
        output_value = output.get_value()
        if isinstance(output_value, Clarification) and output_value.plan_run_id is None:
            output_value.plan_run_id = run_data.plan_run.id

        output_schema = self.output_schema or tool.structured_output_schema
        if (
            output_schema
            and not isinstance(output_value, output_schema)
            and not is_clarification(output_value)
        ):
            model = run_data.config.get_default_model()
            output_value = await model.aget_structured_response(
                [
                    Message(
                        role="user",
                        content=(
                            f"The following was the output from a call to the tool '{tool.id}' "
                            f"with args '{args}': {output}. Convert this output to the desired "
                            f"schema: {output_schema}"
                        ),
                    )
                ],
                output_schema,
            )
        return output_value

    @override
    def to_step_data(self, plan: Plan) -> StepData:
        """Convert this InvokeToolStep to a legacy Step."""
        args_desc = ", ".join(
            [f"{k}={self._resolve_input_names_for_printing(v, plan)}" for k, v in self.args.items()]
        )
        tool_name = self.tool if isinstance(self.tool, str) else self.tool.id
        return Step(
            task=f"Use tool {tool_name} with args: {args_desc}",
            inputs=self._inputs_to_legacy_plan_variables(list(self.args.values()), plan),
            tool_id=tool_name,
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
            condition=self._get_legacy_condition(plan),
        )
