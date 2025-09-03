"""Fluent API for constructing :class:`PlanV2` instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from portia.builder.conditionals import ConditionalBlock, ConditionalBlockClauseType
from portia.builder.plan_v2 import PlanV2
from portia.builder.reference import default_step_name
from portia.builder.step_v2 import (
    ConditionalStep,
    InvokeToolStep,
    LLMStep,
    SingleToolAgentStep,
    StepV2,
    UserInputStep,
    UserVerifyStep,
)
from portia.plan import PlanInput
from portia.telemetry.telemetry_service import ProductTelemetry
from portia.telemetry.views import PlanV2BuildTelemetryEvent
from portia.tool_decorator import tool

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pydantic import BaseModel

    from portia.common import Serializable
    from portia.tool import Tool


class PlanBuilderError(ValueError):
    """Error in Plan definition."""


class PlanBuilderV2:
    """Chainable builder used to assemble Portia plans."""

    def __init__(self, label: str = "Run the plan built with the Plan Builder") -> None:
        """Initialize the builder.

        Args:
            label: Human readable label shown in the Portia dashboard.

        """
        self.plan = PlanV2(steps=[], label=label)
        self._conditional_block_stack: list[ConditionalBlock] = []

    def input(
        self,
        *,
        name: str,
        description: str | None = None,
        default_value: Any | None = None,  # noqa: ANN401
    ) -> PlanBuilderV2:
        """Add an input required by the plan.

        Args:
            name: Name of the input.
            description: Optional description shown to users.
            default_value: Optional default value.

        Returns:
            Self for fluent chaining.
        """
        self.plan.plan_inputs.append(
            PlanInput(name=name, description=description, value=default_value)
        )
        return self

    @property
    def _current_conditional_block(self) -> ConditionalBlock | None:
        """Return the current conditional block if one is active."""
        return self._conditional_block_stack[-1] if len(self._conditional_block_stack) > 0 else None

    def if_(
        self,
        condition: Callable[..., bool] | str,
        args: dict[str, Any] | None = None,
    ) -> PlanBuilderV2:
        """Start a new conditional block.

        Subsequent steps are executed only when the ``condition`` evaluates to
        ``True``. Close the block with :py:meth:`endif`.
        """
        parent_block = self._current_conditional_block
        conditional_block = ConditionalBlock(
            clause_step_indexes=[len(self.plan.steps)],
            parent_conditional_block=parent_block,
        )
        self._conditional_block_stack.append(conditional_block)
        self.plan.steps.append(
            ConditionalStep(
                condition=condition,
                args=args or {},
                step_name=default_step_name(len(self.plan.steps)),
                conditional_block=conditional_block,
                clause_index_in_block=0,
                block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
            )
        )
        return self

    def else_if_(
        self,
        condition: Callable[..., bool],
        args: dict[str, Any] | None = None,
    ) -> PlanBuilderV2:
        """Add an ``else if`` clause to the current conditional block."""
        if len(self._conditional_block_stack) == 0:
            raise PlanBuilderError(
                "else_if_ must be called from a conditional block. Please add an if_ first."
            )
        self._conditional_block_stack[-1].clause_step_indexes.append(len(self.plan.steps))
        self.plan.steps.append(
            ConditionalStep(
                condition=condition,
                args=args or {},
                step_name=default_step_name(len(self.plan.steps)),
                conditional_block=self._conditional_block_stack[-1],
                clause_index_in_block=len(self._conditional_block_stack[-1].clause_step_indexes)
                - 1,
                block_clause_type=ConditionalBlockClauseType.ALTERNATE_CLAUSE,
            )
        )
        return self

    def else_(self) -> PlanBuilderV2:
        """Add an ``else`` clause to the current conditional block."""
        if len(self._conditional_block_stack) == 0:
            raise PlanBuilderError(
                "else_ must be called from a conditional block. Please add an if_ first."
            )
        self._conditional_block_stack[-1].clause_step_indexes.append(len(self.plan.steps))
        self.plan.steps.append(
            ConditionalStep(
                condition=lambda: True,
                args={},
                step_name=default_step_name(len(self.plan.steps)),
                conditional_block=self._conditional_block_stack[-1],
                clause_index_in_block=len(self._conditional_block_stack[-1].clause_step_indexes)
                - 1,
                block_clause_type=ConditionalBlockClauseType.ALTERNATE_CLAUSE,
            )
        )
        return self

    def endif(self) -> PlanBuilderV2:
        """Close the most recently opened conditional block."""
        if len(self._conditional_block_stack) == 0:
            raise PlanBuilderError(
                "endif must be called from a conditional block. Please add an if_ first."
            )
        self._conditional_block_stack[-1].clause_step_indexes.append(len(self.plan.steps))
        self.plan.steps.append(
            ConditionalStep(
                condition=lambda: True,
                args={},
                step_name=default_step_name(len(self.plan.steps)),
                conditional_block=self._conditional_block_stack[-1],
                clause_index_in_block=len(self._conditional_block_stack[-1].clause_step_indexes)
                - 1,
                block_clause_type=ConditionalBlockClauseType.END_CONDITION_BLOCK,
            )
        )
        self._conditional_block_stack.pop()
        return self

    def llm_step(
        self,
        *,
        task: str,
        inputs: list[Any] | None = None,
        output_schema: type[BaseModel] | None = None,
        step_name: str | None = None,
        system_prompt: str | None = None,
    ) -> PlanBuilderV2:
        """Add a step that sends a task to the underlying LLM.

        Args:
            task: Instruction given to the LLM.
            inputs: Optional additional context for the LLM. Values may reference
                previous step outputs or plan inputs.
            output_schema: Expected schema of the result.
            step_name: Optional explicit name; auto-generated if omitted.
            system_prompt: Optional system prompt for the LLM.

        Returns:
            Self for fluent chaining.
        """
        self.plan.steps.append(
            LLMStep(
                task=task,
                inputs=inputs or [],
                output_schema=output_schema,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_block=self._current_conditional_block,
                system_prompt=system_prompt,
            )
        )
        return self

    def invoke_tool_step(
        self,
        *,
        tool: str | Tool,
        args: dict[str, Any] | None = None,
        output_schema: type[BaseModel] | None = None,
        step_name: str | None = None,
    ) -> PlanBuilderV2:
        """Add a step that invokes a tool directly.

        Args:
            tool: Tool id, ``Tool`` instance, or callable to invoke.
            args: Arguments passed to the tool. References are resolved at runtime.
            output_schema: Expected schema of the result.
            step_name: Optional explicit name; auto-generated if omitted.

        Returns:
            Self for fluent chaining.
        """
        self.plan.steps.append(
            InvokeToolStep(
                tool=tool,
                args=args or {},
                output_schema=output_schema,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_block=self._current_conditional_block,
            )
        )
        return self

    def function_step(
        self,
        *,
        function: Callable[..., Any],
        args: dict[str, Any] | None = None,
        output_schema: type[BaseModel] | None = None,
        step_name: str | None = None,
    ) -> PlanBuilderV2:
        """Add a step that calls a Python function.

        Args:
            function: Function to invoke.
            args: Arguments passed to the function. References are resolved at
                runtime.
            output_schema: Expected schema of the result.
            step_name: Optional explicit name; auto-generated if omitted.

        Returns:
            Self for fluent chaining.
        """
        tool_class = tool(function)
        return self.invoke_tool_step(
            tool=tool_class(),
            args=args,
            output_schema=output_schema,
            step_name=step_name,
        )

    def single_tool_agent_step(
        self,
        *,
        tool: str,
        task: str,
        inputs: list[Any] | None = None,
        output_schema: type[BaseModel] | None = None,
        step_name: str | None = None,
    ) -> PlanBuilderV2:
        """Add a step where an agent uses a single tool to complete ``task``.

        Args:
            tool: Tool the agent may call once.
            task: Natural language description of the goal.
            inputs: Optional context for the agent. Values may reference prior
                step outputs or plan inputs.
            output_schema: Expected schema of the result.
            step_name: Optional explicit name; auto-generated if omitted.

        Returns:
            Self for fluent chaining.
        """
        self.plan.steps.append(
            SingleToolAgentStep(
                tool=tool,
                task=task,
                inputs=inputs or [],
                output_schema=output_schema,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_block=self._current_conditional_block,
            )
        )
        return self

    def user_verify(self, *, message: str, step_name: str | None = None) -> PlanBuilderV2:
        """Add a user confirmation step.

        The user must accept ``message`` for the plan to continue.

        Args:
            message: Text shown to the user. May include references to previous
                steps or inputs.
            step_name: Optional explicit name; auto-generated if omitted.

        Returns:
            Self for fluent chaining.
        """
        self.plan.steps.append(
            UserVerifyStep(
                message=message,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_block=self._current_conditional_block,
            )
        )
        return self

    def user_input(
        self,
        *,
        message: str,
        options: list[Serializable] | None = None,
        step_name: str | None = None,
    ) -> PlanBuilderV2:
        """Add a step that requests input from the user.

        Depending on whether ``options`` are provided, the user is prompted with
        free text or multiple choice input.

        Args:
            message: Guidance shown to the user. May include references to
                previous steps or inputs.
            options: Choices for multiple choice prompts. ``None`` produces a
                free-text input.
            step_name: Optional explicit name; auto-generated if omitted.

        Returns:
            Self for fluent chaining.
        """
        self.plan.steps.append(
            UserInputStep(
                message=message,
                options=options,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_block=self._current_conditional_block,
            )
        )
        return self

    def add_step(self, step: StepV2) -> PlanBuilderV2:
        """Add a pre-built step to the plan.

        Useful for integrating custom step types.
        """
        self.plan.steps.append(step)
        return self

    def add_steps(
        self,
        plan: PlanV2 | Iterable[StepV2],
        input_values: dict[str, Any] | None = None,
    ) -> PlanBuilderV2:
        """Add multiple steps or merge another plan into this builder.

        Args:
            plan: Iterable of steps or a ``PlanV2`` to merge.
            input_values: Optional mapping of input names to default values when
                ``plan`` is a ``PlanV2``.

        Returns:
            Self for fluent chaining.

        Raises:
            PlanBuilderError: If duplicate inputs are detected.
        """
        if isinstance(plan, PlanV2):
            # Ensure there are no duplicate plan inputs
            existing_input_names = {p.name for p in self.plan.plan_inputs}
            for _input in plan.plan_inputs:
                if _input.name in existing_input_names:
                    raise PlanBuilderError(f"Duplicate input {_input.name} found in plan.")
            self.plan.plan_inputs.extend(plan.plan_inputs)
            self.plan.steps.extend(plan.steps)
        else:
            self.plan.steps.extend(plan)

        if input_values and isinstance(plan, PlanV2):
            allowed_input_names = {p.name for p in plan.plan_inputs}
            for input_name, input_value in input_values.items():
                if input_name not in allowed_input_names:
                    raise PlanBuilderError(
                        f"Tried to provide value for input {input_name} not found in "
                        "sub-plan passed into add_steps()."
                    )
                for plan_input in self.plan.plan_inputs:
                    if plan_input.name == input_name:
                        plan_input.value = input_value
                        break

        return self

    def final_output(
        self,
        output_schema: type[BaseModel] | None = None,
        summarize: bool = False,
    ) -> PlanBuilderV2:
        """Define the final output of the plan.

        Args:
            output_schema: Schema for the final output. If provided, an LLM will
                coerce the output to match.
            summarize: Whether to also return a summary of the final output.

        Returns:
            Self for fluent chaining.
        """
        self.plan.final_output_schema = output_schema
        self.plan.summarize = summarize
        return self

    def build(self) -> PlanV2:
        """Finalize and return the built plan."""
        if len(self._conditional_block_stack) > 0:
            raise PlanBuilderError(
                "An endif must be called for all if_ steps. Please add an endif for all if_ steps."
            )

        step_type_counts: dict[str, int] = {}
        for step in self.plan.steps:
            step_type = step.__class__.__name__
            step_type_counts[step_type] = step_type_counts.get(step_type, 0) + 1

        telemetry = ProductTelemetry()
        telemetry.capture(
            PlanV2BuildTelemetryEvent(
                plan_length=len(self.plan.steps), step_type_counts=step_type_counts
            )
        )

        return self.plan
