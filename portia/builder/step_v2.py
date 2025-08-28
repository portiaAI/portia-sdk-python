"""Interface for steps that are run as part of a PlanV2."""

from __future__ import annotations

import inspect
import itertools
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, override

from langsmith import traceable
from pydantic import BaseModel, ConfigDict, Field, field_validator

from portia.builder.conditionals import (
    ConditionalBlock,
    ConditionalBlockClauseType,
    ConditionalStepResult,
)
from portia.builder.reference import Input, Reference, ReferenceValue, StepOutput
from portia.clarification import (
    Clarification,
    ClarificationCategory,
    ClarificationType,
    InputClarification,
    MultipleChoiceClarification,
    UserVerificationClarification,
)
from portia.errors import PlanRunExitError, ToolNotFoundError
from portia.execution_agents.conditional_evaluation_agent import ConditionalEvaluationAgent
from portia.execution_agents.execution_utils import is_clarification
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
    conditional_block: ConditionalBlock | None = Field(
        default=None, description="The conditional block this step is part of, if any."
    )

    @abstractmethod
    async def run(self, run_data: RunContext) -> Any:  # noqa: ANN401
        """Execute the step."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this step to a Step from plan.py.

        A Step is the legacy representation of a step in the plan, and is still used in the
        Portia backend. If this step doesn't need to be represented in the plan sent to the Portia
        backend, return None.
        """
        raise NotImplementedError  # pragma: no cover

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

    def _get_legacy_condition(self, plan: PlanV2) -> str | None:
        """Get the legacy condition for a step."""
        if self.conditional_block is None:
            return None
        step_names = [s.step_name for s in plan.steps]
        current_step_index = step_names.index(self.step_name)

        def get_conditional_for_nested_block(block: ConditionalBlock) -> str | None:
            active_clause_step_index = next(
                itertools.dropwhile(
                    # First clause step index where the current step index is greater
                    # than the clause step index e.g. for clause step indexes [1, 8, 12]
                    # and current step index 2, the active clause step index is 1
                    lambda x: current_step_index < x,
                    reversed(block.clause_step_indexes),
                ),
                None,
            )
            if active_clause_step_index is None:
                raise ValueError(f"Cannot determine active conditional for step {self.step_name}")

            if (
                current_step_index == block.clause_step_indexes[0]
                or current_step_index == block.clause_step_indexes[-1]
            ):
                # The step is the `if_` or the `endif` step, so no new condition is needed
                # as this will always be evaluated at this 'depth' of the plan branching.
                return None

            # All previous clause conditions must be false for this step to get run
            previous_clause_step_indexes = itertools.takewhile(
                lambda x: x < current_step_index,
                itertools.filterfalse(
                    lambda x: x == active_clause_step_index, block.clause_step_indexes
                ),
            )
            condition_str = " and ".join(
                f"{plan.step_output_name(i)} is false" for i in previous_clause_step_indexes
            )
            if current_step_index not in block.clause_step_indexes:
                # The step is a non-conditional step within a block, so we need to make the
                # active clause condition was true.
                condition_str = f"{plan.step_output_name(active_clause_step_index)} is true" + (
                    f" and {condition_str}" if condition_str else ""
                )

            return condition_str

        legacy_condition_strings = []
        current_block = self.conditional_block
        while current_block is not None:
            legacy_condition_string = get_conditional_for_nested_block(current_block)
            if legacy_condition_string is not None:
                legacy_condition_strings.append(legacy_condition_string)
            current_block = current_block.parent_conditional_block
        return "If " + " and ".join(legacy_condition_strings) if legacy_condition_strings else None


class LLMStep(StepV2):
    """A step that runs a given task through an LLM (without any tools)."""

    task: str = Field(description="The task to perform.")
    inputs: list[Any] = Field(
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
    system_prompt: str | None = Field(
        default=None,
        description=(
            "The prompt to use for the LLM. If not provided, uses default prompt from LLMTool."
        ),
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        return f"LLMStep(task='{self.task}'{output_info})"

    @override
    @traceable(name="LLM Step - Run")
    async def run(self, run_data: RunContext) -> str | BaseModel:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the LLM query."""
        if self.system_prompt:
            llm_tool = LLMTool(
                structured_output_schema=self.output_schema, prompt=self.system_prompt
            )
        else:
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
            condition=self._get_legacy_condition(plan),
        )


class InvokeToolStep(StepV2):
    """A step that calls a tool with the given args (no LLM involved, just a direct tool call)."""

    tool: str | Tool = Field(
        description=(
            "The tool to use. Should either be the id of the tool to run or the Tool instance to "
            "run."
        )
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The args to call the tool with. The arg values can be references to previous step "
            "outputs / plan inputs (using StepOutput / Input) or just plain values."
        ),
    )
    output_schema: type[BaseModel] | None = Field(
        default=None, description="The schema of the output."
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        return f"InvokeToolStep(tool='{self._tool_name()}', args={self.args}{output_info})"

    def _tool_name(self) -> str:
        """Get the name of the tool."""
        if isinstance(self.tool, str):
            return self.tool
        return self.tool.id

    @override
    @traceable(name="Invoke Tool Step - Run")
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
            clarifications=run_data.plan_run.get_clarifications_for_step(
                run_data.plan_run.current_step_index
            ),
        )
        args = {k: self._get_value_for_input(v, run_data) for k, v in self.args.items()}

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
            model = run_data.portia.config.get_default_model()
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
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this InvokeToolStep to a legacy Step."""
        inputs_desc = ", ".join(
            [f"{k}={self._resolve_input_names_for_printing(v, plan)}" for k, v in self.args.items()]
        )
        return Step(
            task=f"Use tool {self._tool_name()} with inputs: {inputs_desc}",
            inputs=self._inputs_to_legacy_plan_variables(list(self.args.values()), plan),
            tool_id=self._tool_name(),
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
            condition=self._get_legacy_condition(plan),
        )


class FunctionStep(StepV2):
    """Calls a function with the given args (no LLM involved, just a direct function call).

    The function can be either synchronous or asynchronous. Async functions will be properly
    awaited.
    """

    function: Callable[..., Any] = Field(description=("The function to call."))
    args: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The args to call the function with. The arg values can be references to previous step "
            "outputs / plan inputs (using StepOutput / Input) or just plain values."
        ),
    )
    output_schema: type[BaseModel] | None = Field(
        default=None, description="The schema of the output."
    )

    _TOOL_ID_PREFIX: ClassVar[str] = "local_function_"

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        fn_name = getattr(self.function, "__name__", str(self.function))
        return f"FunctionStep(function='{fn_name}', args={self.args}{output_info})"

    @override
    @traceable(name="Function Step - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the function."""
        args = {k: self._get_value_for_input(v, run_data) for k, v in self.args.items()}

        if inspect.iscoroutinefunction(self.function):
            output = await self.function(**args)
        else:
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
                        content=(
                            f"The following was the output from a call to the function "
                            f"'{self.function.__name__}' with args '{args}': {output}. Convert "
                            f"this output to the desired schema: {self.output_schema}"
                        ),
                    )
                ],
                self.output_schema,
            )
        return output

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this FunctionStep to a legacy Step."""
        inputs_desc = ", ".join(
            [f"{k}={self._resolve_input_names_for_printing(v, plan)}" for k, v in self.args.items()]
        )
        fn_name = getattr(self.function, "__name__", str(self.function))
        return Step(
            task=f"Run function {fn_name} with args: {inputs_desc}",
            inputs=self._inputs_to_legacy_plan_variables(list(self.args.values()), plan),
            tool_id=f"{self._TOOL_ID_PREFIX}{fn_name}",
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
            condition=self._get_legacy_condition(plan),
        )

    @classmethod
    def tool_id_is_local_function(cls, tool_id: str) -> bool:
        """Check if the tool id is a local function."""
        return tool_id.startswith(cls._TOOL_ID_PREFIX)


class SingleToolAgentStep(StepV2):
    """A step where an LLM agent uses a single tool (calling it only once) to complete a task."""

    task: str = Field(description="The task to perform.")
    tool: str = Field(description="The tool to use.")
    inputs: list[Any] = Field(
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

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        return f"SingleToolAgentStep(tool='{self.tool}', query='{self.task}'{output_info})"

    @override
    @traceable(name="Single Tool Agent Step - Run")
    async def run(self, run_data: RunContext) -> None:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the agent step."""
        agent = run_data.portia.get_agent_for_step(
            self.to_legacy_step(run_data.plan), run_data.legacy_plan, run_data.plan_run
        )
        output_obj = await agent.execute_async()
        return output_obj.get_value()

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this SingleToolAgentStep to a Step."""
        return Step(
            task=self.task,
            inputs=self._inputs_to_legacy_plan_variables(self.inputs, plan),
            tool_id=self.tool,
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
            condition=self._get_legacy_condition(plan),
        )


class UserVerifyStep(StepV2):
    """A step that asks the user to verify a message before continuing."""

    message: str = Field(description="The message the user needs to verify.")

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        return f"UserVerifyStep(message='{self.message}')"

    @override
    @traceable(name="User Verify Step - Run")
    async def run(self, run_data: RunContext) -> bool | UserVerificationClarification:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Run the user verification step."""
        message = self._resolve_input_reference(self.message, run_data)

        previous_clarification = run_data.plan_run.get_clarification_for_step(
            ClarificationCategory.USER_VERIFICATION
        )

        if not previous_clarification or not previous_clarification.resolved:
            return UserVerificationClarification(
                plan_run_id=run_data.plan_run.id,
                user_guidance=str(message),
                source="User verify step",
            )

        if previous_clarification.response is False:
            raise PlanRunExitError(f"User rejected verification: {message}")

        return True

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this UserVerifyStep to a legacy Step."""
        return Step(
            task=f"User verification: {self.message}",
            inputs=[],
            tool_id=None,
            output=plan.step_output_name(self),
            condition=self._get_legacy_condition(plan),
        )


class UserInputStep(StepV2):
    """A step that requests input from the user and returns the response.

    If options are provided, creates a multiple choice clarification.
    Otherwise, creates a text input clarification.
    """

    message: str = Field(description="The guidance message shown to the user.")
    options: list[Any] | None = Field(
        default=None,
        description="Available options for multiple choice. If None, creates text input.",
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        input_type = "multiple choice" if self.options else "text input"
        return f"UserInputStep(type='{input_type}', message='{self.message}')"

    def _create_clarification(self, run_data: RunContext) -> ClarificationType:
        """Create the appropriate clarification based on whether options are provided."""
        resolved_message = self._resolve_input_reference(self.message, run_data)

        if self.options:
            return MultipleChoiceClarification(
                plan_run_id=run_data.plan_run.id,
                user_guidance=str(resolved_message),
                options=self.options,
                argument_name=run_data.plan.step_output_name(self),
                source="User input step",
            )
        return InputClarification(
            plan_run_id=run_data.plan_run.id,
            user_guidance=str(resolved_message),
            argument_name=run_data.plan.step_output_name(self),
            source="User input step",
        )

    @override
    @traceable(name="User Input Step - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Run the user input step."""
        clarification_type = (
            ClarificationCategory.MULTIPLE_CHOICE if self.options else ClarificationCategory.INPUT
        )

        previous_clarification = run_data.plan_run.get_clarification_for_step(clarification_type)

        if not previous_clarification or not previous_clarification.resolved:
            return self._create_clarification(run_data)

        return previous_clarification.response

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this UserInputStep to a legacy Step."""
        input_type = "Multiple choice" if self.options else "Text input"
        return Step(
            task=f"User input ({input_type}): {self.message}",
            inputs=[],
            tool_id=None,
            output=plan.step_output_name(self),
            condition=self._get_legacy_condition(plan),
        )


class ConditionalStep(StepV2):
    """A step that represents a conditional clause in a conditional block.

    I.E. if, else-if, else, end-if clauses.
    """

    condition: Callable[..., bool] | str = Field(
        description=(
            "The boolean predicate to check. If evaluated to true, the steps within this clause "
            "will be evaluated - otherwise they will be skipped and we jump to the next clause."
        )
    )
    args: dict[str, Reference | Any] = Field(
        default_factory=dict, description="The args to check the condition with."
    )
    clause_index_in_block: int = Field(description="The index of the clause in the condition block")
    block_clause_type: ConditionalBlockClauseType

    @field_validator("conditional_block", mode="after")
    @classmethod
    def validate_conditional_block(cls, v: ConditionalBlock | None) -> ConditionalBlock:
        """Validate the conditional block."""
        if v is None:
            raise ValueError("Conditional block is required for ConditionSteps")
        return v

    @property
    def block(self) -> ConditionalBlock:
        """Get the conditional block for this step."""
        if not isinstance(self.conditional_block, ConditionalBlock):
            raise TypeError("Conditional block is not a ConditionalBlock")
        return self.conditional_block

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        return (
            f"ConditionalStep(condition='{self.condition}', "
            f"clause_type='{self.block_clause_type.value}' args={self.args})"
        )

    @override
    @traceable(name="Conditional Step - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the conditional step."""
        args = {k: self._get_value_for_input(v, run_data) for k, v in self.args.items()}
        if isinstance(self.condition, str):
            condition_str = self._get_value_for_input(self.condition, run_data)
            agent = ConditionalEvaluationAgent(run_data.portia.config)
            conditional_result = await agent.execute(condition_str, args)
        else:
            conditional_result = self.condition(**args)
        next_clause_step_index = (
            self.block.clause_step_indexes[self.clause_index_in_block + 1]
            if self.clause_index_in_block < len(self.block.clause_step_indexes) - 1
            else self.block.clause_step_indexes[self.clause_index_in_block]
        )
        return ConditionalStepResult(
            type=self.block_clause_type,
            conditional_result=conditional_result,
            next_clause_step_index=next_clause_step_index,
            end_condition_block_step_index=self.block.clause_step_indexes[-1],
        )

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this ConditionalStep to a PlanStep."""
        if isinstance(self.condition, str):
            cond_str = self.condition
        else:
            cond_str = (
                "If result of "
                + getattr(self.condition, "__name__", str(self.condition))
                + " is true"
            )
        return Step(
            task=f"Conditional clause: {cond_str}",
            inputs=self._inputs_to_legacy_plan_variables(list(self.args.values()), plan),
            tool_id=None,
            output=plan.step_output_name(self),
            condition=self._get_legacy_condition(plan),
        )
