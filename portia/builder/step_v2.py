"""Interface for steps that are run as part of a PlanV2."""

from __future__ import annotations

import itertools
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, override

from langsmith import traceable
from pydantic import BaseModel, ConfigDict, Field, field_validator

from portia.builder.conditionals import (
    ConditionalBlock,
    ConditionalBlockClauseType,
    ConditionalStepResult,
)
from portia.builder.reference import Input, Reference, StepOutput
from portia.clarification import (
    Clarification,
    ClarificationCategory,
    ClarificationType,
    InputClarification,
    MultipleChoiceClarification,
    UserVerificationClarification,
)
from portia.config import ExecutionAgentType
from portia.errors import PlanRunExitError, ToolNotFoundError
from portia.execution_agents.conditional_evaluation_agent import ConditionalEvaluationAgent
from portia.execution_agents.default_execution_agent import DefaultExecutionAgent
from portia.execution_agents.execution_utils import is_clarification
from portia.execution_agents.one_shot_agent import OneShotAgent
from portia.execution_agents.output import LocalDataValue
from portia.logger import logger
from portia.model import Message
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import PlanInput, Step, Variable
from portia.tool import Tool, ToolRunContext
from portia.tool_wrapper import ToolCallWrapper

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2
    from portia.execution_agents.base_execution_agent import BaseExecutionAgent
    from portia.run_context import RunContext


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
        value: Any | Reference,  # noqa: ANN401
        run_data: RunContext,
    ) -> Any | None:  # noqa: ANN401
        """Resolve any references in the provided value to their actual values.

        If value is a Reference (e.g. Input or StepOutput), then the value that Reference refers to
        is returned. If the value is a string with a Reference in it, then the string is returned
        with the reference values templated in. Any other value is returned unchanged.
        """
        if isinstance(value, Reference):
            value = value.get_value(run_data)
            return self._resolve_input_reference(value, run_data)
        if isinstance(value, str):
            return self._template_input_references(value, run_data)
        return value

    def _template_input_references(self, value: str, run_data: RunContext) -> str:
        # Extract all instances of {{ StepOutput(var_name) }} or {{ Input(var_name) }}
        # from _input if it's a string
        matches = re.findall(r"\{\{\s*(StepOutput|Input)\s*\(\s*([\w\s]+)\s*\)\s*\}\}", value)

        # If there are matches, replace each {{ StepOutput(var_name) }}
        # or {{ Input(var_name) }} with its resolved value.
        if matches:
            result = value
            for ref_type, var_name in matches:
                var_name = var_name.strip()  # noqa: PLW2901
                if ref_type == "StepOutput" and var_name.isdigit():
                    var_name = int(var_name)  # noqa: PLW2901
                ref = StepOutput(var_name) if ref_type == "StepOutput" else Input(var_name)  # type: ignore reportArgumentType
                resolved = self._resolve_input_reference(ref, run_data)
                pattern = (
                    r"\{\{\s*"
                    + re.escape(ref_type)
                    + r"\s*\(\s*"
                    + re.escape(str(var_name))
                    + r"\s*\)\s*\}\}"
                )
                result = re.sub(pattern, str(resolved), result, count=1)
            return result
        return value

    def _resolve_input_names_for_printing(
        self,
        _input: Any,  # noqa: ANN401
        plan: PlanV2,
    ) -> Any | None:  # noqa: ANN401
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
                raise ValueError(
                    f"Cannot determine active conditional for step {self.step_name}"
                )  # pragma: no cover

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
        wrapped_tool = ToolCallWrapper(
            child_tool=llm_tool,
            storage=run_data.storage,
            plan_run=run_data.plan_run,
        )
        tool_ctx = ToolRunContext(
            end_user=run_data.end_user,
            plan_run=run_data.plan_run,
            plan=run_data.legacy_plan,
            config=run_data.config,
            clarifications=[],
        )
        task_data = []
        for _input in self.inputs:
            if isinstance(_input, Reference):
                description = self._get_ref_description(_input, run_data)
                value = self._resolve_input_reference(_input, run_data)
                value = LocalDataValue(value=value, summary=description)
            else:
                value = self._resolve_input_reference(_input, run_data)
            if value is not None or not isinstance(_input, Reference):
                task_data.append(value)

        return await wrapped_tool.arun(tool_ctx, task=self.task, task_data=task_data)

    def _get_ref_description(self, ref: Reference, run_data: RunContext) -> str:
        """Get the description of a reference."""
        if isinstance(ref, StepOutput):
            return ref.get_description(run_data)
        if isinstance(ref, Input):
            plan_input = self._plan_input_from_name(ref.name, run_data)
            if plan_input.description:
                return plan_input.description
            if isinstance(plan_input.value, Reference):
                return self._get_ref_description(plan_input.value, run_data)
        return ""

    def _plan_input_from_name(self, name: str, run_data: RunContext) -> PlanInput:
        """Get the plan input from the name."""
        for plan_input in run_data.plan.plan_inputs:
            if plan_input.name == name:
                return plan_input
        raise ValueError(f"Plan input {name} not found")

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

        tool_ctx = ToolRunContext(
            end_user=run_data.end_user,
            plan_run=run_data.plan_run,
            plan=run_data.legacy_plan,
            config=run_data.config,
            clarifications=run_data.plan_run.get_clarifications_for_step(
                run_data.plan_run.current_step_index
            ),
        )
        args = {k: self._resolve_input_reference(v, run_data) for k, v in self.args.items()}
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
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this InvokeToolStep to a legacy Step."""
        args_desc = ", ".join(
            [f"{k}={self._resolve_input_names_for_printing(v, plan)}" for k, v in self.args.items()]
        )
        return Step(
            task=f"Use tool {self._tool_name()} with args: {args_desc}",
            inputs=self._inputs_to_legacy_plan_variables(list(self.args.values()), plan),
            tool_id=self._tool_name(),
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
            condition=self._get_legacy_condition(plan),
        )


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
        agent = self._get_agent_for_step(run_data)
        output_obj = await agent.execute_async()
        return output_obj.get_value()

    def _get_agent_for_step(
        self,
        run_data: RunContext,
    ) -> BaseExecutionAgent:
        """Get the appropriate agent for executing the step."""
        tool = ToolCallWrapper.from_tool_id(
            self.tool,
            run_data.tool_registry,
            run_data.storage,
            run_data.plan_run,
        )
        cls: type[BaseExecutionAgent]
        match run_data.config.execution_agent_type:
            case ExecutionAgentType.ONE_SHOT:
                cls = OneShotAgent
            case ExecutionAgentType.DEFAULT:
                cls = DefaultExecutionAgent
        cls = OneShotAgent if isinstance(tool, LLMTool) else cls
        logger().debug(
            f"Using agent: {type(cls).__name__}",
            plan=str(run_data.plan.id),
            plan_run=str(run_data.plan_run.id),
        )
        return cls(
            run_data.legacy_plan,
            run_data.plan_run,
            run_data.config,
            run_data.storage,
            run_data.end_user,
            tool,
            execution_hooks=run_data.execution_hooks,
        )

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
        message = self._template_input_references(self.message, run_data)

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
        resolved_message = self._template_input_references(self.message, run_data)

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
        args = {k: self._resolve_input_reference(v, run_data) for k, v in self.args.items()}
        if isinstance(self.condition, str):
            condition_str = self._template_input_references(self.condition, run_data)
            agent = ConditionalEvaluationAgent(run_data.config)
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
