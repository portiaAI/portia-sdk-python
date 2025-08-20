"""Interface for steps that are run as part of a PlanV2."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, override

from langsmith import traceable
from pydantic import BaseModel, ConfigDict, Field

from portia.builder.reference import Input, Reference, ReferenceValue, StepOutput
from portia.clarification import Clarification
from portia.errors import ToolNotFoundError
from portia.model import Message
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import Step, Variable
from portia.tool import Tool, ToolRunContext

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2
    from portia.portia import RunContext


class StepV2(BaseModel, ABC):
    """Interface for steps that are run as part of a plan."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_name: str = Field(description="The name of the step.")

    @abstractmethod
    async def run(self, run_data: RunContext) -> Any:  # noqa: ANN401
        """Execute the step."""
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> str:
        """Return a description of this step for logging purposes."""
        raise NotImplementedError

    @abstractmethod
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this step to a Step from plan.py.

        A Step is the legacy representation of a step in the plan, and is still used in the
        Portia backend. If this step doesn't need to be represented in the plan sent to the Portia
        backend, return None.
        """
        raise NotImplementedError

    def _resolve_input_reference(
        self,
        _input: Any,  # noqa: ANN401
        run_data: RunContext,
    ) -> Any | ReferenceValue | None:  # noqa: ANN401
        """Resolve input values by retrieving the ReferenceValue for any Reference inputs."""
        if isinstance(_input, str):
            # Extract all instances of {{ StepOutput(var_name) }} or {{ Input(var_name) }}
            # from _input if it's a string
            matches = re.findall(r"\{\{\s*(StepOutput|Input)\s*\(\s*([\w\s]+)\s*\)\s*\}\}", _input)

            # If there are matches, replace each {{ StepOutput(var_name) }}
            # or {{ Input(var_name) }} with its resolved value.
            if isinstance(_input, str) and matches:
                result = _input
                for ref_type, var_name in matches:
                    var_name = var_name.strip()  # noqa: PLW2901
                    if ref_type == "StepOutput" and var_name.isdigit():
                        var_name = int(var_name)  # noqa: PLW2901
                    ref = StepOutput(var_name) if ref_type == "StepOutput" else Input(var_name)  # type: ignore reportArgumentType
                    resolved = self._resolve_input_reference(ref, run_data)
                    resolved_val = (
                        resolved.value.full_value(run_data.portia.storage)
                        if isinstance(resolved, ReferenceValue)
                        else resolved
                    )
                    pattern = (
                        r"\{\{\s*"
                        + re.escape(ref_type)
                        + r"\s*\(\s*"
                        + re.escape(str(var_name))
                        + r"\s*\)\s*\}\}"
                    )
                    result = re.sub(pattern, str(resolved_val), result, count=1)
                return result
        return _input.get_value(run_data) if isinstance(_input, Reference) else _input

    def _get_value_for_input(self, _input: Any, run_data: RunContext) -> Any | None:  # noqa: ANN401
        """Get the value for an input that could come from a reference."""
        resolved_input = self._resolve_input_reference(_input, run_data)

        if isinstance(resolved_input, ReferenceValue):
            return resolved_input.value.full_value(run_data.portia.storage)
        return resolved_input

    def _resolve_input_names_for_printing(
        self,
        _input: Any,  # noqa: ANN401
        plan: PlanV2,
    ) -> Any | ReferenceValue | None:  # noqa: ANN401
        """Resolve inputs to their value (if not a reference) or to their name (if reference).

        Useful for printing inputs before the plan is run.
        """
        if isinstance(_input, Reference):
            name = _input.get_legacy_name(plan)
            # Ensure name starts with a $ so that it is clear it is a reference
            # This is done so it appears nicely in the UI
            if not name.startswith("$"):
                name = f"${name}"
            return name
        if isinstance(_input, list):
            return [self._resolve_input_names_for_printing(v, plan) for v in _input]
        return _input

    def _inputs_to_legacy_plan_variables(self, inputs: list[Any], plan: PlanV2) -> list[Variable]:
        """Convert a list of inputs to a list of legacy plan variables."""
        return [Variable(name=v.get_legacy_name(plan)) for v in inputs if isinstance(v, Reference)]


class LLMStep(StepV2):
    """A step that runs a given task through an LLM (without any tools)."""

    task: str = Field(description="The task to perform.")
    inputs: list[Any | Reference] = Field(
        default_factory=list,
        description=(
            "The inputs for the task. The inputs can be references to previous step outputs / "
            "plan inputs (using StepOutput / Input) or just plain values. They are passed in as "
            "additional context to the LLM when it is completing the task."
        ),
    )
    output_schema: type[BaseModel] | None = Field(
        default=None, description="The schema of the output."
    )

    @override
    def describe(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        return f"LLMStep(task='{self.task}'{output_info})"

    @override
    @traceable(name="LLM Step - Run")
    async def run(self, run_data: RunContext) -> str | BaseModel:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the LLM query."""
        llm_tool = LLMTool(structured_output_schema=self.output_schema)
        tool_ctx = ToolRunContext(
            end_user=run_data.end_user,
            plan_run=run_data.plan_run,
            plan=run_data.legacy_plan,
            config=run_data.portia.config,
            clarifications=[],
        )
        task_data = [
            self._format_value(value, run_data)
            for _input in self.inputs or []
            if (value := self._resolve_input_reference(_input, run_data)) is not None
            or not isinstance(_input, Reference)
        ]
        return await llm_tool.arun(tool_ctx, task=self.task, task_data=task_data)

    def _format_value(self, _input: Any, run_data: RunContext) -> Any | None:  # noqa: ANN401
        """Get the value for an input."""
        if not isinstance(_input, ReferenceValue):
            return _input
        return (
            f"Previous step {_input.description} had output: "
            f"{_input.value.full_value(run_data.portia.storage)}"
        )

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this LLMStep to a Step."""
        return Step(
            task=self.task,
            inputs=self._inputs_to_legacy_plan_variables(self.inputs, plan),
            tool_id=LLMTool.LLM_TOOL_ID,
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
        )


class ToolRun(StepV2):
    """A step that calls a tool with the given args (no LLM involved, just a direct tool call)."""

    tool: str | Tool = Field(
        description=(
            "The tool to use. Should either be the id of the tool to run or the Tool instance to "
            "run."
        )
    )
    args: dict[str, Any | Reference] = Field(
        default_factory=dict,
        description=(
            "The args to call the tool with. The arg values can be references to previous step "
            "outputs / plan inputs (using StepOutput / Input) or just plain values."
        ),
    )
    output_schema: type[BaseModel] | None = Field(
        default=None, description="The schema of the output."
    )

    @override
    def describe(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        return f"ToolRun(tool='{self._tool_name()}', args={self.args}{output_info})"

    def _tool_name(self) -> str:
        """Get the name of the tool."""
        if isinstance(self.tool, str):
            return self.tool
        return self.tool.id

    @override
    @traceable(name="Tool Run - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the tool."""
        if isinstance(self.tool, str):
            tool = run_data.portia.get_tool(self.tool, run_data.plan_run)
        else:
            tool = self.tool
        if not tool:
            raise ToolNotFoundError(self.tool if isinstance(self.tool, str) else self.tool.id)

        tool_ctx = ToolRunContext(
            end_user=run_data.end_user,
            plan_run=run_data.plan_run,
            plan=run_data.legacy_plan,
            config=run_data.portia.config,
            clarifications=[],
        )
        args = {k: self._get_value_for_input(v, run_data) for k, v in self.args.items()}

        # TODO(RH): Move to async tool run when we can  # noqa: FIX002, TD003
        output = tool.run(tool_ctx, **args)
        if isinstance(output, Clarification) and output.plan_run_id is None:
            output.plan_run_id = run_data.plan_run.id

        if (
            self.output_schema
            and not isinstance(output, self.output_schema)
            and not isinstance(output, Clarification)
        ):
            model = run_data.portia.config.get_default_model()
            output = await model.aget_structured_response(
                [
                    Message(
                        role="user",
                        content=f"Convert this output to the desired schema: {output}",
                    )
                ],
                self.output_schema,
            )
        return output

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this ToolCall to a Step."""
        inputs_desc = ", ".join(
            [f"{k}={self._resolve_input_names_for_printing(v, plan)}" for k, v in self.args.items()]
        )
        return Step(
            task=f"Use tool {self._tool_name()} with inputs: {inputs_desc}",
            inputs=self._inputs_to_legacy_plan_variables(list(self.args.values()), plan),
            tool_id=self._tool_name(),
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
        )


class FunctionCall(StepV2):
    """Calls a function with the given args (no LLM involved, just a direct function call)."""

    function: Callable[..., Any] = Field(description=("The function to call."))
    args: dict[str, Any | Reference] = Field(
        default_factory=dict,
        description=(
            "The args to call the function with. The arg values can be references to previous step "
            "outputs / plan inputs (using StepOutput / Input) or just plain values."
        ),
    )
    output_schema: type[BaseModel] | None = Field(
        default=None, description="The schema of the output."
    )

    @override
    def describe(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        fn_name = getattr(self.function, "__name__", str(self.function))
        return f"FunctionCall(function='{fn_name}', args={self.args}{output_info})"

    @override
    @traceable(name="Function Call - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the function."""
        args = {k: self._get_value_for_input(v, run_data) for k, v in self.args.items()}
        output = self.function(**args)

        if isinstance(output, Clarification) and output.plan_run_id is None:
            output.plan_run_id = run_data.plan_run.id

        if (
            self.output_schema
            and not isinstance(output, self.output_schema)
            and not isinstance(output, Clarification)
        ):
            model = run_data.portia.config.get_default_model()
            output = await model.aget_structured_response(
                [
                    Message(
                        role="user",
                        content=f"Convert this output to the desired schema: {output}",
                    )
                ],
                self.output_schema,
            )
        return output

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this ToolCall to a Step."""
        inputs_desc = ", ".join(
            [f"{k}={self._resolve_input_names_for_printing(v, plan)}" for k, v in self.args.items()]
        )
        fn_name = getattr(self.function, "__name__", str(self.function))
        return Step(
            task=f"Run function {fn_name} with args: {inputs_desc}",
            inputs=self._inputs_to_legacy_plan_variables(list(self.args.values()), plan),
            tool_id=f"local_function_{fn_name}",
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
        )


class SingleToolAgent(StepV2):
    """A step where an LLM agent uses a single tool (calling it only once) to complete a task."""

    task: str = Field(description="The task to perform.")
    tool: str = Field(description="The tool to use.")
    inputs: list[Any | Reference] = Field(
        default_factory=list,
        description=(
            "The inputs for the task. The inputs can be references to previous step outputs / "
            "plan inputs (using StepOutput / Input) or just plain values. They are passed in as "
            "additional context to the agent when it is completing the task."
        ),
    )
    output_schema: type[BaseModel] | None = Field(
        default=None, description="The schema of the output."
    )

    @override
    def describe(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        return f"SingleToolAgent(tool='{self.tool}', query='{self.task}'{output_info})"

    @override
    @traceable(name="Single Tool Agent - Run")
    async def run(self, run_data: RunContext) -> None:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the agent step."""
        agent = run_data.portia.get_agent_for_step(
            self.to_legacy_step(run_data.plan), run_data.legacy_plan, run_data.plan_run
        )
        output_obj = await agent.execute_async()
        return output_obj.get_value()

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this SingleToolAgent to a Step."""
        return Step(
            task=self.task,
            inputs=self._inputs_to_legacy_plan_variables(self.inputs, plan),
            tool_id=self.tool,
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
        )
