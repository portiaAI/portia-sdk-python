"""Interface for steps that are run as part of a PortiaPlan."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, override

from langsmith import traceable
from pydantic import BaseModel, Field

from portia.builder.reference import Reference, ReferenceValue
from portia.model import Message
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import Step as PlanStep
from portia.plan import Variable
from portia.tool import ToolRunContext
from portia.tool_wrapper import ToolCallWrapper

if TYPE_CHECKING:
    from portia.builder.portia_plan import PortiaPlan
    from portia.portia import RunData


class Step(BaseModel, ABC):
    """Interface for steps that are run as part of a plan."""

    name: str = Field(description="The name of the step.")

    @abstractmethod
    async def run(self, run_data: RunData) -> Any:  # noqa: ANN401
        """Execute the step."""
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> str:
        """Return a description of this step for logging purposes."""
        raise NotImplementedError

    @abstractmethod
    def to_legacy_step(self, plan: PortiaPlan) -> PlanStep:
        """Convert this step to a PlanStep from plan.py.

        A PlanStep is the legacy representation of a step in the plan, and is still used in the
        Portia backend. If this step doesn't need to be represented in the plan sent to the Portia
        backend, return None.
        """
        raise NotImplementedError

    def _resolve_input_reference(
        self,
        _input: Any,  # noqa: ANN401
        run_data: RunData,
    ) -> Any | ReferenceValue | None:  # noqa: ANN401
        """Resolve input values by retrieving the ReferenceValue for any Reference inputs."""
        return _input.get_value(run_data) if isinstance(_input, Reference) else _input

    def _get_value_for_input(self, _input: Any, run_data: RunData) -> Any | None:  # noqa: ANN401
        """Get the value for an input that could come from a reference."""
        resolved_input = self._resolve_input_reference(_input, run_data)

        if isinstance(resolved_input, ReferenceValue):
            return resolved_input.value.full_value(run_data.portia.storage)
        return resolved_input

    def _resolve_input_names_for_printing(
        self,
        _input: Any,  # noqa: ANN401
        plan: PortiaPlan,
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

    def _inputs_to_legacy_plan_variables(
        self, inputs: list[Any], plan: PortiaPlan
    ) -> list[Variable]:
        """Convert a list of inputs to a list of legacy plan variables."""
        return [Variable(name=v.get_legacy_name(plan)) for v in inputs if isinstance(v, Reference)]


class LLMStep(Step):
    """A step that runs a given task through an LLM (without any tools)."""

    task: str = Field(description="The task to perform.")
    inputs: list[Any] = Field(default_factory=list, description="The inputs to the task.")
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
    async def run(self, run_data: RunData) -> str | BaseModel:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
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

    def _format_value(self, _input: Any, run_data: RunData) -> Any | None:  # noqa: ANN401
        """Get the value for an input."""
        if not isinstance(_input, ReferenceValue):
            return _input
        return (
            f"Previous step {_input.description} had output: "
            f"{_input.value.full_value(run_data.portia.storage)}"
        )

    @override
    def to_legacy_step(self, plan: PortiaPlan) -> PlanStep:
        """Convert this LLMStep to a PlanStep."""
        return PlanStep(
            task=self.task,
            inputs=self._inputs_to_legacy_plan_variables(self.inputs, plan),
            tool_id=LLMTool.LLM_TOOL_ID,
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
        )


class ToolCall(Step):
    """A step that calls a tool with the given inputs."""

    tool: str = Field(description="The id of the tool to call.")
    args: dict[str, Any] = Field(
        default_factory=dict, description="The args to call the tool with."
    )
    output_schema: type[BaseModel] | None = Field(
        default=None, description="The schema of the output."
    )

    @override
    def describe(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        return f"ToolCall(tool='{self.tool}', inputs={self.args}{output_info})"

    @override
    @traceable(name="Tool Call - Run")
    async def run(self, run_data: RunData) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the tool."""
        tool = run_data.portia.tool_registry.get_tool(self.tool)
        wrapped_tool = ToolCallWrapper(
            child_tool=tool,
            storage=run_data.portia.storage,
            plan_run=run_data.plan_run,
        )
        tool_ctx = ToolRunContext(
            end_user=run_data.end_user,
            plan_run=run_data.plan_run,
            plan=run_data.legacy_plan,
            config=run_data.portia.config,
            clarifications=[],
        )
        args = {k: self._get_value_for_input(v, run_data) for k, v in self.args.items()}

        # TODO(RH): Move to async tool run when we can  # noqa: FIX002, TD003
        output = wrapped_tool.run(tool_ctx, **args)
        if self.output_schema and not isinstance(output, self.output_schema):
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
    def to_legacy_step(self, plan: PortiaPlan) -> PlanStep:
        """Convert this ToolCall to a PlanStep."""
        inputs_desc = ", ".join(
            [f"{k}={self._resolve_input_names_for_printing(v, plan)}" for k, v in self.args.items()]
        )
        return PlanStep(
            task=f"Use tool {self.tool} with inputs: {inputs_desc}",
            inputs=self._inputs_to_legacy_plan_variables(list(self.args.values()), plan),
            tool_id=self.tool,
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
        )


class FunctionCall(Step):
    """A step that calls a function with the given inputs."""

    function: Callable[..., Any] = Field(description="The function to call.")
    args: dict[str, Any] = Field(
        default_factory=dict, description="The args to call the function with."
    )
    output_schema: type[BaseModel] | None = Field(
        default=None, description="The schema of the output."
    )

    @override
    def describe(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        return f"FunctionCall(function='{self.function.__name__}', inputs={self.args}{output_info})"

    @override
    @traceable(name="Function Call - Run")
    async def run(self, run_data: RunData) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the function."""
        args = {k: self._get_value_for_input(v, run_data) for k, v in self.args.items()}
        output = self.function(**args)

        if self.output_schema and not isinstance(output, self.output_schema):
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
    def to_legacy_step(self, plan: PortiaPlan) -> PlanStep:
        """Convert this FunctionCall to a PlanStep."""
        inputs_desc = ", ".join(
            [f"{k}={self._resolve_input_names_for_printing(v, plan)}" for k, v in self.args.items()]
        )
        return PlanStep(
            task=f"Call function {self.function.__name__} with inputs: {inputs_desc}",
            inputs=self._inputs_to_legacy_plan_variables(list(self.args.values()), plan),
            tool_id=f"local_function_{self.function.__name__}",
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
        )


class SingleToolAgent(Step):
    """A step where an LLM agent uses a single tool (calling it only once) to complete a task."""

    task: str = Field(description="The task to perform.")
    tool: str = Field(description="The tool to use.")
    inputs: list[Any] = Field(default_factory=list, description="The inputs to the tool.")
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
    async def run(self, run_data: RunData) -> None:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the agent step."""
        agent = run_data.portia.get_agent_for_step(
            self.to_legacy_step(run_data.plan), run_data.legacy_plan, run_data.plan_run
        )
        output_obj = await agent.execute_async()
        return output_obj.get_value()

    @override
    def to_legacy_step(self, plan: PortiaPlan) -> PlanStep:
        """Convert this SingleToolAgent to a PlanStep."""
        return PlanStep(
            task=self.task,
            inputs=self._inputs_to_legacy_plan_variables(self.inputs, plan),
            tool_id=self.tool,
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
        )


class Hook(Step):
    """A custom function that runs between steps to extend / modify the running of the plan."""

    hook: Callable[..., None] = Field(description="The hook to run.")
    args: dict[str, Any] = Field(default_factory=dict)

    @override
    def describe(self) -> str:
        """Return a description of this step for logging purposes."""
        hook_name = getattr(self.hook, "__name__", str(self.hook))
        return f"Hook(hook={hook_name}, args={self.args})"

    @override
    @traceable(name="Hook - Run")
    async def run(self, run_data: RunData) -> None:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the hook."""
        args = {k: self._get_value_for_input(v, run_data) for k, v in self.args.items()}
        return self.hook(**args)

    @override
    def to_legacy_step(self, plan: PortiaPlan) -> PlanStep:
        """Convert this Hook to a PlanStep."""
        hook_name = getattr(self.hook, "__name__", str(self.hook))
        return PlanStep(
            task=f"Run hook: {hook_name}",
            inputs=self._inputs_to_legacy_plan_variables(list(self.args.values()), plan),
            tool_id=f"local_hook_{hook_name}",
            output=plan.step_output_name(self),
            structured_output_schema=None,
        )
