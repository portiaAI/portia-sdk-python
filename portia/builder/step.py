"""Interface for steps that are run as part of a PortiaPlan."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, override

from langsmith import traceable
from pydantic import BaseModel, Field

from portia.builder.reference import Reference, ReferenceValue
from portia.execution_agents.default_execution_agent import DefaultExecutionAgent
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

    name: str

    @abstractmethod
    async def run(self, run_data: RunData) -> Any:  # noqa: ANN401
        """Execute the step."""
        raise NotImplementedError

    @abstractmethod
    def describe(self, run_data: RunData) -> str:
        """Return a description of this step for logging purposes."""
        raise NotImplementedError

    @abstractmethod
    def to_portia_step(self, plan: PortiaPlan) -> PlanStep:
        """Convert this step to a PlanStep from plan.py.

        A PlanStep is the legacy representation of a step in the plan, and is still used in the
        Portia backend. If this step doesn't need to be represented in the plan sent to the Portia
        backend, return None.
        """
        raise NotImplementedError

    def _get_value_for_input(self, _input: Any, run_data: RunData) -> Any | ReferenceValue | None:  # noqa: ANN401
        """Get the value for an input that could come from a previous step output."""
        return _input.get_value(run_data) if isinstance(_input, Reference) else _input


class LLMStep(Step):
    """A step that runs a given task through an LLM (without any tools)."""

    task: str
    inputs: list[Any] | None = None
    output_schema: type[BaseModel] | None = None

    @property
    def tool_id(self) -> str:
        """Return the LLM tool ID."""
        return LLMTool.LLM_TOOL_ID

    @override
    def describe(self, run_data: RunData) -> str:
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
            if (value := self._get_value_for_input(_input, run_data)) is not None
            or not isinstance(_input, Reference)
        ]
        return await llm_tool.arun(tool_ctx, task=self.task, task_data=task_data)

    def _format_value(self, _input: Any, run_data: RunData) -> Any | None:  # noqa: ANN401
        """Get the value for an input."""
        if not isinstance(_input, ReferenceValue):
            return _input
        step_output_value: ReferenceValue = _input
        return (
            f"Previous step {step_output_value.description} had output: "
            f"{step_output_value.value.full_value(run_data.portia.storage)}"
        )

    @override
    def to_portia_step(self, plan: PortiaPlan) -> PlanStep:
        """Convert this LLMStep to a PlanStep."""
        input_variables = [
            Variable(name=v.get_name(plan)) for v in self.inputs or [] if isinstance(v, Reference)
        ]
        return PlanStep(
            task=self.task,
            inputs=input_variables,
            tool_id=LLMTool.LLM_TOOL_ID,
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
        )


class ToolCall(Step):
    """A step that calls a tool with the given inputs."""

    tool: str
    args: dict[str, Any]
    output_schema: type[BaseModel] | None = None

    @override
    def describe(self, run_data: RunData) -> str:
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
        args = {
            k: (
                value.value.full_value(run_data.portia.storage)
                if isinstance(value, ReferenceValue)
                else value
            )
            for k, v in self.args.items()
            if (value := self._get_value_for_input(v, run_data)) is not None
            or not isinstance(v, Reference)
        }

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
    def to_portia_step(self, plan: PortiaPlan) -> PlanStep:
        """Convert this ToolCall to a PlanStep."""
        inputs_desc = ", ".join([f"{k}={v}" for k, v in self.args.items()])
        input_variables = [
            Variable(name=v.get_name(plan)) for v in self.args.values() if isinstance(v, Reference)
        ]
        return PlanStep(
            task=f"Use tool {self.tool} with inputs: {inputs_desc}",
            inputs=input_variables,
            tool_id=self.tool,
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
        )


class FunctionCall(Step):
    """A step that calls a function with the given inputs."""

    function: Callable[..., Any]
    args: dict[str, Any]
    output_schema: type[BaseModel] | None = None

    @override
    def describe(self, run_data: RunData) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        return f"FunctionCall(function='{self.function.__name__}', inputs={self.args}{output_info})"

    @override
    @traceable(name="Function Call - Run")
    async def run(self, run_data: RunData) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the function."""
        args = {
            k: (
                value.value.full_value(run_data.portia.storage)
                if isinstance(value, ReferenceValue)
                else value
            )
            for k, v in self.args.items()
            if (value := self._get_value_for_input(v, run_data)) is not None
            or not isinstance(v, Reference)
        }
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
    def to_portia_step(self, plan: PortiaPlan) -> PlanStep:
        """Convert this FunctionCall to a PlanStep."""
        inputs_desc = ", ".join([f"{k}={v}" for k, v in self.args.items()])
        input_variables = [
            Variable(name=v.get_name(plan)) for v in self.args.values() if isinstance(v, Reference)
        ]
        return PlanStep(
            task=f"Call function {self.function.__name__} with inputs: {inputs_desc}",
            inputs=input_variables,
            tool_id=f"local_function_{self.function.__name__}",
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
        )


class SingleToolAgent(Step):
    """A step where an LLM agent uses a single tool (calling it only once) to complete a task."""

    task: str
    tool: str
    inputs: list[Any] | None = None
    output_schema: type[BaseModel] | None = None

    @override
    def describe(self, run_data: RunData) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        return f"SingleToolAgent(tool='{self.tool}', query='{self.task}'{output_info})"

    @override
    @traceable(name="Single Tool Agent - Run")
    async def run(self, run_data: RunData) -> None:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the agent step."""
        tool = run_data.portia.tool_registry.get_tool(self.tool)
        wrapped_tool = ToolCallWrapper(
            child_tool=tool,
            storage=run_data.portia.storage,
            plan_run=run_data.plan_run,
        )
        agent = DefaultExecutionAgent(
            plan=run_data.legacy_plan,
            plan_run=run_data.plan_run,
            config=run_data.portia.config,
            agent_memory=run_data.portia.storage,
            end_user=run_data.end_user,
            tool=wrapped_tool,
        )
        output_obj = await agent.execute_async()
        return output_obj.get_value()

    @override
    def to_portia_step(self, plan: PortiaPlan) -> PlanStep:
        """Convert this SingleToolAgent to a PlanStep."""
        input_variables = [
            Variable(name=v.get_name(plan)) for v in self.inputs or [] if isinstance(v, Reference)
        ]
        return PlanStep(
            task=self.task,
            inputs=input_variables,
            tool_id=self.tool,
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
        )


class Hook(Step):
    """A custom function that runs between steps to extend / modify the running of the plan."""

    hook: Callable[..., None]
    args: dict[str, Any] = Field(default_factory=dict)

    @override
    def describe(self, run_data: RunData) -> str:
        """Return a description of this step for logging purposes."""
        hook_name = getattr(self.hook, "__name__", str(self.hook))
        return f"Hook(hook={hook_name}, args={self.args})"

    @override
    @traceable(name="Hook - Run")
    async def run(self, run_data: RunData) -> None:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the hook."""
        args = {
            k: (
                value.value.full_value(run_data.portia.storage)
                if isinstance(value, ReferenceValue)
                else value
            )
            for k, v in self.args.items()
            if (value := self._get_value_for_input(v, run_data)) is not None
            or not isinstance(v, Reference)
        }
        return self.hook(**args)

    @override
    def to_portia_step(self, plan: PortiaPlan) -> PlanStep:
        """Convert this Hook to a PlanStep."""
        input_variables = [
            Variable(name=v.get_name(plan)) for v in self.args.values() if isinstance(v, Reference)
        ]
        return PlanStep(
            task="Run hook",
            inputs=input_variables,
            tool_id=f"local_hook_{self.hook.__name__}",
            output=plan.step_output_name(self),
            structured_output_schema=None,
        )
