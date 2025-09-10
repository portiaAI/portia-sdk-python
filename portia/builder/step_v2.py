"""Implementation of the various step types used in :class:`PlanV2`."""

from __future__ import annotations

import itertools
import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Self

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override  # pragma: no cover

from langsmith import traceable
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from portia.builder.conditionals import (
    ConditionalBlock,
    ConditionalBlockClauseType,
    ConditionalStepResult,
)
from portia.builder.loops import LoopBlock, LoopStepResult, LoopStepType, LoopType
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
from portia.execution_agents.react_agent import ReActAgent
from portia.logger import logger
from portia.model import Message
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import PlanInput, Step, Variable
from portia.tool import Tool
from portia.tool_wrapper import ToolCallWrapper

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2
    from portia.execution_agents.base_execution_agent import BaseExecutionAgent
    from portia.run_context import RunContext


class StepV2(BaseModel, ABC):
    """Abstract base class for all steps executed within a plan.

    Each step represents an action that can be performed during plan execution,
    such as calling an LLM / agent, invoking a tool, or requesting user input.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_name: str = Field(description="Unique name identifying this step within the plan.")
    conditional_block: ConditionalBlock | None = Field(
        default=None,
        description="The conditional block containing this step, if part of conditional logic.",
    )
    loop_block: LoopBlock | None = Field(
        default=None, description="The loop block this step is part of, if any."
    )

    @abstractmethod
    async def run(self, run_data: RunContext) -> Any | LocalDataValue:  # noqa: ANN401
        """Execute the step and return its output.

        Returns:
            The step's output value, which may be used by subsequent steps.

        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this step to the legacy Step format.

        This is primarily used to determine how the steps should be presented in the Portia
        Dashboard.
        """
        raise NotImplementedError  # pragma: no cover

    def _resolve_references(
        self,
        value: Any | Reference,  # noqa: ANN401
        run_data: RunContext,
    ) -> Any | None:  # noqa: ANN401
        """Resolve any Reference objects to their concrete values.

        This method handles 3 types of value:
        * A Reference object - this will be resolved to its concrete value
        * A string containing ``{{ StepOutput(...) }}`` or ``{{ Input(...) }}`` templates - these
          will be rendered with the referenced values
        * Any other value - this will be returned unchanged
        """
        if isinstance(value, Reference):
            value = value.get_value(run_data)
            return self._resolve_references(value, run_data)
        if isinstance(value, str):
            return self._template_references(value, run_data)
        return value

    def _resolve_input_references_with_descriptions(
        self, inputs: list[Any], run_data: RunContext
    ) -> list[LocalDataValue | Any]:
        """Resolve all references in a list of inputs, including descriptions.

        For each value in inputs, if value is a Reference (e.g. Input or StepOutput), then the value
        that Reference refers to is returned, in a LocalDataValue alongside a description. If the
        value is a string with a Reference in it, then the string is returned with the reference
        values templated in. Any other value is returned unchanged.

        This method is primarily used to provide the inputs as additional information LLMs.
        """
        resolved_inputs = []
        for _input in inputs:
            if isinstance(_input, Reference):
                description = self._get_ref_description(_input, run_data)
                value = self._resolve_references(_input, run_data)
                value = LocalDataValue(value=value, summary=description)
            else:
                value = self._resolve_references(_input, run_data)
            if value is not None or not isinstance(_input, Reference):
                resolved_inputs.append(value)
        return resolved_inputs

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
        raise ValueError(f"Plan input {name} not found")  # pragma: no cover

    def _template_references(self, value: str, run_data: RunContext) -> str:
        """Replace any Reference objects in a string with their resolved values.

        For example, if the string is f"The result was {StepOutput(0)}", and the step output
        value is "step result", then the string will be replaced with "The result was step result".

        Supports the following reference types:
        - {{ StepOutput(step_name) }}
        - {{ StepOutput('step_name', path='field.name') }}
        - {{ Input(input_name) }}
        """
        # Find all {{ ... }} blocks that contain StepOutput or Input
        pattern = r"\{\{\s*(StepOutput|Input)\s*\([^}]+\)\s*\}\}"

        def replace_reference(match: re.Match[str]) -> str:
            full_match = match.group(0)
            ref_type_str = match.group(1)

            # Extract the content inside the parentheses
            # e.g., from "{{ StepOutput('step', path='field.name') }}"
            # get "'step', path='field.name'"
            paren_content = full_match[full_match.find("(") + 1 : full_match.rfind(")")]

            try:
                ref_cls = StepOutput if ref_type_str == "StepOutput" else Input
                ref_obj = self._parse_reference_expression(ref_cls, paren_content)
                resolved = self._resolve_references(ref_obj, run_data)
                return str(resolved)
            except (ValueError, AttributeError, KeyError):  # pragma: no cover
                # If parsing fails, return the original match
                return full_match  # pragma: no cover

        return re.sub(pattern, replace_reference, value)

    def _parse_reference_expression(
        self, ref_cls: type[Reference], paren_content: str
    ) -> Reference:
        """Parse the content inside StepOutput(...) or Input(...) to create the reference object.

        Args:
            ref_cls: The Reference class to instantiate (StepOutput or Input)
            paren_content: The content inside the parentheses

        Supports the following reference types:
        - StepOutput(step_name)
        - StepOutput(step_name, path='field.name')
        - Input(input_name)

        """
        paren_content = paren_content.strip()

        if paren_content.isdigit() and ref_cls == StepOutput:
            # Reference with a step index - e.g. StepOutput(0)
            return StepOutput(int(paren_content))

        if ", path=" in paren_content:
            return self._parse_reference_with_path(ref_cls, paren_content)

        # Simple reference with no path - e.g. StepOutput("step_name") or Input("input_name")
        if (paren_content.startswith('"') and paren_content.endswith('"')) or (
            paren_content.startswith("'") and paren_content.endswith("'")
        ):
            name = paren_content[1:-1]  # Remove quotes
            if ref_cls == StepOutput:
                return StepOutput(name)
            return Input(name)

        raise ValueError(f"Invalid reference format: {paren_content}")  # pragma: no cover

    def _parse_reference_with_path(self, ref_cls: type[Reference], paren_content: str) -> Reference:
        """Parse reference expressions that include path parameters.

        Args:
            ref_cls: The Reference class to instantiate (StepOutput or Input)
            paren_content: The content inside the parentheses

        Handles cases like:
        - 'step_name', path='field.name'
        - 42, path='field.name'  (numeric step index)

        Step/input names must be quoted strings, except for numeric step indices.

        """
        # Extract path parameter - should always be present since we detected a comma
        path_match = re.search(r"\bpath\s*=\s*['\"]([^'\"]*)['\"]", paren_content)
        if not path_match:
            raise ValueError(  # pragma: no cover
                f"Expected path parameter in reference expression: {paren_content}"
            )
        path_param = path_match.group(1)

        # Extract the first parameter (step/input name)
        # Split by comma and take the first part
        first_param = paren_content.split(",")[0].strip()

        if first_param.isdigit():
            step_param = int(first_param)
        elif (first_param.startswith('"') and first_param.endswith('"')) or (
            first_param.startswith("'") and first_param.endswith("'")
        ):
            step_param = first_param[1:-1]  # Remove quotes
        else:
            raise ValueError(  # pragma: no cover
                f"Expected quoted string or number for step/input name, got: {first_param}"
            )

        if ref_cls == StepOutput:
            return StepOutput(step_param, path=path_param)
        return Input(str(step_param), path=path_param)

    def _resolve_input_names_for_printing(
        self,
        _input: Any,  # noqa: ANN401
        plan: PlanV2,
    ) -> Any | None:  # noqa: ANN401
        """Resolve any References in the provided input to their name (note: not to their value).

        For example, StepOutput(0) will be resolved to "step_0_output", not the concrete value it
        represents. This is useful for printing inputs before the plan is run.
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
    """A step that executes a task using an LLM without any tool access.

    This step is used for pure language model tasks like text generation,
    analysis, or transformation that don't require external tool calls.
    """

    task: str = Field(description="The natural language task for the LLM to perform.")
    inputs: list[Any] = Field(
        default_factory=list,
        description=(
            "Additional context data for the task. Can include references to previous step "
            "outputs (using StepOutput) or plan inputs (using Input), or literal values. "
            "These are provided as context to help the LLM complete the task."
        ),
    )
    output_schema: type[BaseModel] | None = Field(
        default=None,
        description=(
            "Pydantic model class defining the expected structure of the LLM's response. "
            "If provided, the output from the LLM will be coerced to match this schema."
        ),
    )
    system_prompt: str | None = Field(
        default=None,
        description=(
            "Custom system prompt to guide the LLM's behavior. If not specified, "
            "the default LLMTool system prompt will be used."
        ),
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        return f"LLMStep(task='{self.task}'{output_info})"

    @override
    @traceable(name="LLM Step - Run")
    async def run(self, run_data: RunContext) -> str | BaseModel:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Execute the LLM task and return its response."""
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

        tool_ctx = run_data.get_tool_run_ctx()
        task_data = self._resolve_input_references_with_descriptions(self.inputs, run_data)
        task = self._template_references(self.task, run_data)
        return await wrapped_tool.arun(tool_ctx, task=task, task_data=task_data)

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this LLMStep to a legacy Step."""
        return Step(
            task=self.task,
            inputs=self._inputs_to_legacy_plan_variables(self.inputs, plan),
            tool_id=LLMTool.LLM_TOOL_ID,
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
            condition=self._get_legacy_condition(plan),
        )


class InvokeToolStep(StepV2):
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
    def to_legacy_step(self, plan: PlanV2) -> Step:
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


class SingleToolAgentStep(StepV2):
    """A step where an LLM agent intelligently uses a specific tool to complete a task.

    Unlike InvokeToolStep which requires you to specify exact tool arguments, this step
    allows an LLM agent to determine how to use the tool based on the task description
    and available context. The agent will call the tool at most once during execution.
    """

    task: str = Field(description="Natural language description of the task to accomplish.")
    tool: str | Tool = Field(
        description=(
            "ID of the tool the agent should use to complete the task or the Tool instance itself."
        )
    )
    inputs: list[Any] = Field(
        default_factory=list,
        description=(
            "Additional context data for the agent. Can include references to previous step "
            "outputs (using StepOutput), plan inputs (using Input), or literal values. "
            "The agent will use this context to determine how to call the tool."
        ),
    )
    output_schema: type[BaseModel] | None = Field(
        default=None,
        description=(
            "Pydantic model class defining the expected structure of the agent's output. "
            "If provided, the output from the agent will be coerced to match this schema."
        ),
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        tool_name = self.tool if isinstance(self.tool, str) else self.tool.id
        return f"SingleToolAgentStep(task='{self.task}', tool='{tool_name}'{output_info})"

    @override
    @traceable(name="Single Tool Agent Step - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the agent and return its output."""
        agent = self._get_agent_for_step(run_data)
        return await agent.execute_async()

    def _get_agent_for_step(
        self,
        run_data: RunContext,
    ) -> BaseExecutionAgent:
        """Get the appropriate agent for executing the step."""
        if isinstance(self.tool, str):
            tool = ToolCallWrapper.from_tool_id(
                self.tool,
                run_data.tool_registry,
                run_data.storage,
                run_data.plan_run,
            )
        else:
            if self.tool.id not in run_data.tool_registry:
                run_data.tool_registry.with_tool(self.tool)
            tool = ToolCallWrapper(
                child_tool=self.tool,
                storage=run_data.storage,
                plan_run=run_data.plan_run,
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
            tool_id=self.tool if isinstance(self.tool, str) else self.tool.id,
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
            condition=self._get_legacy_condition(plan),
        )


class ReActAgentStep(StepV2):
    """A step where an LLM agent uses ReAct reasoning to complete a task with multiple tools.

    Unlike SingleToolAgentStep which is limited to one specific tool and one tool call, this step
    allows an LLM agent to reason about which tools to use and when to use them. The agent
    follows the ReAct (Reasoning and Acting) pattern, iteratively thinking about the
    problem and taking actions until the task is complete.
    """

    task: str = Field(description="Natural language description of the task to accomplish.")
    tools: Sequence[str | Tool] = Field(
        description=(
            "IDs of the tools the agent can use to complete the task or Tool instances themselves."
        )
    )
    inputs: list[Any] = Field(
        default_factory=list,
        description=(
            "The inputs for the task. The inputs can be references to previous step outputs / "
            "plan inputs (using StepOutput / Input) or just plain values. They are passed in as "
            "additional context to the agent when it is completing the task."
        ),
    )
    output_schema: type[BaseModel] | None = Field(
        default=None,
        description=(
            "Pydantic model class defining the expected structure of the agent's output. "
            "If provided, the output from the agent will be coerced to match this schema."
        ),
    )
    tool_call_limit: int = Field(
        default=25,
        description="The maximum number of tool calls to make before the agent stops.",
    )
    allow_agent_clarifications: bool = Field(
        default=False,
        description=(
            "Whether to allow the agent to ask clarifying questions to the user "
            "if it is unable to proceed. When set to true, the agent can output clarifications "
            "that can be resolved by the user to get input. In order to use this, make sure you "
            "clarification handler set up that is capable of handling InputClarifications."
        ),
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        tools_list = [t if isinstance(t, str) else t.id for t in self.tools]
        return f"ReActAgentStep(task='{self.task}', tools='{tools_list}', {output_info})"

    @override
    @traceable(name="ReAct Agent Step - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the agent step."""
        agent = self._get_agent_for_step(run_data)
        return await agent.execute()

    def _get_agent_for_step(
        self,
        run_data: RunContext,
    ) -> ReActAgent:
        """Get the appropriate agent for executing the step."""
        tools = []
        for tool in self.tools:
            if isinstance(tool, str):
                tool_wrapper = ToolCallWrapper.from_tool_id(
                    tool, run_data.tool_registry, run_data.storage, run_data.plan_run
                )
            else:
                if tool.id not in run_data.tool_registry:
                    run_data.tool_registry.with_tool(tool)
                tool_wrapper = ToolCallWrapper(
                    child_tool=tool,
                    storage=run_data.storage,
                    plan_run=run_data.plan_run,
                )
            if tool_wrapper is not None:
                tools.append(tool_wrapper)
        task = self._template_references(self.task, run_data)
        task_data = self._resolve_input_references_with_descriptions(self.inputs, run_data)

        return ReActAgent(
            task=task,
            task_data=task_data,
            tools=tools,
            run_data=run_data,
            tool_call_limit=self.tool_call_limit,
            allow_agent_clarifications=self.allow_agent_clarifications,
            output_schema=self.output_schema,
        )

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this SingleToolAgentStep to a Step."""
        return Step(
            task=self.task,
            inputs=self._inputs_to_legacy_plan_variables(self.inputs, plan),
            tool_id=",".join(tool if isinstance(tool, str) else tool.id for tool in self.tools),
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
            condition=self._get_legacy_condition(plan),
        )


class UserVerifyStep(StepV2):
    """A step that requests user confirmation before proceeding with plan execution.

    This step pauses execution to ask the user to verify or approve a message.
    If the user rejects the verification, the plan execution will stop with an error.

    This pauses plan execution and asks the user to confirm or reject the provided
    message. The plan will only continue if the user confirms. If the user rejects,
    the plan execution will stop with an error. This is useful for getting user approval before
    taking important actions like sending emails, making purchases, or modifying data.

    A UserVerificationClarification is used to get the verification from the user, so ensure you
    have set up handling for this type of clarification in order to use this step. For more
    details, see https://docs.portialabs.ai/understand-clarifications.

    This step outputs True if the user confirms.
    """

    message: str = Field(
        description="The message or action requiring user verification/approval. "
        "It can include references to previous step outputs or plan inputs "
        "(using Input / StepOutput references)"
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        return f"UserVerifyStep(message='{self.message}')"

    @override
    @traceable(name="User Verify Step - Run")
    async def run(self, run_data: RunContext) -> bool | UserVerificationClarification:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Prompt the user for confirmation.

        Returns a UserVerificationClarification to get input from the user (if not already
        provided).

        If the user has already confirmed, returns True. Otherwise, if the user has rejected the
        verification, raises a PlanRunExitError.
        """
        message = self._template_references(self.message, run_data)

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
    """A step that requests input from the user and returns their response.

    This pauses plan execution and prompts the user to provide input. If options are
    provided, the user must choose from the given choices (multiple choice). If no
    options are provided, the user can enter free-form text.

    A Clarification (either InputClarification or MultipleChoiceClarification) is used to get
    the input from the user, so ensure you have set up handling for the required type of
    clarification in order to use this step. For more details, see
    https://docs.portialabs.ai/understand-clarifications.

    The user's response becomes the output of this step and can be referenced by
    subsequent steps in the plan.
    """

    message: str = Field(description="The prompt or question to display to the user.")
    options: list[Any] | None = Field(
        default=None,
        description=(
            "Available choices for multiple-choice input. If provided, the user must select "
            "from these options. If None, allows free-form text input. Options can include "
            "references to previous step outputs or plan inputs (using Input / StepOutput "
            "references)"
        ),
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        input_type = "multiple choice" if self.options else "text input"
        return f"UserInputStep(type='{input_type}', message='{self.message}')"

    def _create_clarification(self, run_data: RunContext) -> ClarificationType:
        """Create the appropriate clarification based on whether options are provided."""
        resolved_message = self._template_references(self.message, run_data)

        if self.options:
            options = [self._resolve_references(o, run_data) for o in self.options]
            return MultipleChoiceClarification(
                plan_run_id=run_data.plan_run.id,
                user_guidance=str(resolved_message),
                options=options,
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
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Request input from the user and return the response."""
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
    """A step that represents a conditional clause within a conditional execution block.

    This step handles conditional logic such as if, else-if, else, and end-if statements
    that control which subsequent steps should be executed based on runtime conditions.
    """

    condition: Callable[..., bool] | str = Field(
        description=(
            "The condition to evaluate for this clause. Can be a callable that returns a boolean, "
            "or a string expression that will be evaluated by an LLM. If true, subsequent "
            "steps in this clause will execute; if false, execution jumps to the next clause."
        )
    )
    args: dict[str, Reference | Any] = Field(
        default_factory=dict,
        description=(
            "Arguments to pass to the condition. Values can be references to step outputs "
            "(using StepOutput), plan inputs (using Input), or literal values."
        ),
    )
    clause_index_in_block: int = Field(
        description="The position of this clause within its conditional block (0-based index)."
    )
    block_clause_type: ConditionalBlockClauseType = Field(
        description="The type of conditional clause (IF, ELIF, ELSE, or ENDIF)."
    )

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
        """Evaluate the condition and return a ConditionalStepResult."""
        args = {k: self._resolve_references(v, run_data) for k, v in self.args.items()}
        if isinstance(self.condition, str):
            condition_str = self._template_references(self.condition, run_data)
            agent = ConditionalEvaluationAgent(run_data.config)
            conditional_result = await agent.execute(conditional=condition_str, arguments=args)
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
        """Convert this ConditionalStep to a legacy Step."""
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


class LoopStep(StepV2):
    """A step that represents a loop in a loop block.

    This step handles loop logic such as while, do-while, and for-each loops that
    control which subsequent steps should be executed based on runtime conditions.
    """

    condition: Callable[..., bool] | str | None = Field(
        description=(
            "The boolean predicate to check. If evaluated to true, the loop will continue."
        )
    )
    over: Reference | Sequence[Any] | None = Field(
        default=None, description="The reference to loop over."
    )
    loop_type: LoopType
    index: int = Field(default=0, description="The current index of the loop.")
    args: dict[str, Reference | Any] = Field(
        default_factory=dict, description="The args to check the condition with."
    )
    loop_step_type: LoopStepType

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        """Validate the loop."""
        if self.condition is None and self.over is None:
            raise ValueError("Condition and over cannot both be None")
        if self.condition is not None and self.loop_type == LoopType.FOR_EACH:
            raise ValueError("Condition cannot be set for for-each loop")
        if self.condition and self.over:
            raise ValueError("Condition and over cannot both be set")
        if self.over is not None and self.loop_type in (LoopType.WHILE, LoopType.DO_WHILE):
            raise ValueError("Over cannot be set for while or do-while loop")
        return self

    def _current_loop_variable(self, run_data: RunContext) -> Any | None:  # noqa: ANN401
        """Get the current loop variable if over is set."""
        if self.over is None:
            return None
        if isinstance(self.over, Sequence):
            values = self.over[self.index]
        else:
            values = self._resolve_references(self.over, run_data)
        if not isinstance(values, Sequence):
            raise TypeError("Loop variable is not indexable")
        try:
            return values[self.index]
        except IndexError:
            return None

    @override
    @traceable(name="Loop Step - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the loop step."""
        args = {k: self._resolve_references(v, run_data) for k, v in self.args.items()}
        match self.loop_step_type, self.loop_type:
            case (LoopStepType.END, LoopType.DO_WHILE) | (LoopStepType.START, LoopType.WHILE):
                return await self._handle_conditional_loop(run_data, args)
            case LoopStepType.START, LoopType.FOR_EACH:
                if self.over is None:
                    raise ValueError("Over is required for for-each loop")
                value = self._current_loop_variable(run_data)
                self.index += 1
                return LoopStepResult(loop_result=value is not None, value=value)
            case _:
                return LoopStepResult(loop_result=True, value=True)

    async def _handle_conditional_loop(
        self, run_data: RunContext, args: dict[str, Any]
    ) -> LoopStepResult:
        if self.condition is None:
            raise ValueError("Condition is required for loop step")
        if isinstance(self.condition, str):
            template_reference = self._resolve_references(self.condition, run_data)
            agent = ConditionalEvaluationAgent(run_data.config)
            conditional_result = await agent.execute(
                conditional=str(template_reference), arguments=args
            )
        else:
            conditional_result = self.condition(**args)
        return LoopStepResult(loop_result=conditional_result, value=conditional_result)

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this LoopStep to a PlanStep."""
        if isinstance(self.condition, str):
            cond_str = self.condition
        else:
            cond_str = (
                "If result of "
                + getattr(self.condition, "__name__", str(self.condition))
                + " is true"
            )
        return Step(
            task=f"Loop clause: {cond_str}",
            inputs=self._inputs_to_legacy_plan_variables(list(self.args.values()), plan),
            tool_id=None,
            output=plan.step_output_name(self),
            condition=self._get_legacy_condition(plan),
        )
