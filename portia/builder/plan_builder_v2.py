"""Builder for Portia plans."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from portia.builder.conditionals import ConditionalBlock, ConditionalBlockClauseType
from portia.builder.plan_v2 import PlanV2
from portia.builder.reference import default_step_name
from portia.builder.step_v2 import (
    ConditionalStep,
    FunctionStep,
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

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pydantic import BaseModel

    from portia.common import Serializable
    from portia.tool import Tool


class PlanBuilderError(ValueError):
    """Error in Plan definition."""


class PlanBuilderV2:
    """Builder for Portia plans."""

    def __init__(self, label: str = "Run the plan built with the Plan Builder") -> None:
        """Initialize the builder.

        Args:
            label: The label of the plan. This is used to identify the plan in the Portia dashboard.

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
        """Add an input to the plan.

        Args:
            name: The name of the input.
            description: The description of the input.
            default_value: The default value of the input.

        """
        self.plan.plan_inputs.append(
            PlanInput(name=name, description=description, value=default_value)
        )
        return self

    @property
    def _current_conditional_block(self) -> ConditionalBlock | None:
        """Get the current conditional block."""
        return self._conditional_block_stack[-1] if len(self._conditional_block_stack) > 0 else None

    def if_(
        self,
        condition: Callable[..., bool] | str,
        args: dict[str, Any] | None = None,
    ) -> PlanBuilderV2:
        """Add a step that checks a condition."""
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
        """Add a step that checks a condition."""
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
        """Add a step that checks a condition."""
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
        """Exit a conditional block."""
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
        """Add a step that sends a query to the underlying LLM.

        Args:
            task: The task to perform.
            inputs: The inputs to the task. The inputs can be references to previous step outputs /
              plan inputs (using StepOutput / Input) or just plain values. They are passed in as
              additional context to the LLM when it is completing the task.
            output_schema: The schema of the output.
            step_name: Optional name for the step. If not provided, will be auto-generated.
            system_prompt: Optional prompt to use for the LLM. If not provided,
                uses default from LLMTool.

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
        """Add a step that directly invokes a tool.

        Args:
            tool: The tool to invoke. Should either be the id of the tool to call, the Tool instance
              to call, or a python function that should be called.
            args: The arguments to the tool. If any of these values are instances of StepOutput or
              Input, the corresponding values will be substituted in when the plan is run.
            output_schema: The schema of the output.
            step_name: Optional name for the step. If not provided, will be auto-generated.

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
        """Add a step that directly invokes a function.

        Args:
            function: The function to invoke.
            args: The arguments to the function. If any of these values are instances of StepOutput
              or Input, the corresponding values will be substituted in when the plan is run.
            output_schema: The schema of the output.
            step_name: Optional name for the step. If not provided, will be auto-generated.

        """
        self.plan.steps.append(
            FunctionStep(
                function=function,
                args=args or {},
                output_schema=output_schema,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_block=self._current_conditional_block,
            )
        )
        return self

    def single_tool_agent_step(
        self,
        *,
        tool: str,
        task: str,
        inputs: list[Any] | None = None,
        output_schema: type[BaseModel] | None = None,
        step_name: str | None = None,
    ) -> PlanBuilderV2:
        """Add a step that uses the execution agent with a tool.

        Args:
            tool: The tool to use.
            task: The task to perform.
            inputs: The inputs to the task. If any of these values are instances of StepOutput or
              Input, the corresponding values will be substituted in when the plan is run.
            output_schema: The schema of the output.
            step_name: Optional name for the step. If not provided, will be auto-generated.

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
        """Add a step that prompts the user to verify the specified message before continuing.

        This step uses a UserVerificationClarification to interact with the user - you must ensure
        you have a clarification handler setup that handles this type of clarification.

        If the user accepts, then the plan will continue. If the user rejects, then this step will
        raise a PlanRunExitError.

        Args:
            message: The message the user needs to verify. You can use inputs / outputs from
              previous steps in this message and the corresponding value will be substituted in when
              the plan is run - e.g.
              message=f"Are you happy to proceed for user '{StepOutput(0)}'?"
            step_name: Optional name for the step. If not provided, will be auto-generated.

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
        """Add a step that requests input from the user and sets the response as the step output.

        This step uses a UserInputClarification / MultipleChoiceClarification (depending on whether
        options are specified) to interact with the user - you must ensure you have a clarification
        handler setup that handles this type of clarification.

        Args:
            message: The guidance message shown to the user. You can use inputs / outputs from
              previous steps in this message and the corresponding value will be substituted in when
              the plan is run - e.g.
              message=f"Enter the value for user '{StepOutput(0)}':"
            options: Available options for multiple choice. If None, creates text input.
            step_name: Optional name for the step. If not provided, will be auto-generated.

        Returns:
            Self for method chaining.

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

        This can be used to add custom steps into the plan.
        """
        self.plan.steps.append(step)
        return self

    def add_steps(
        self,
        plan: PlanV2 | Iterable[StepV2],
        input_values: dict[str, Any] | None = None,
    ) -> PlanBuilderV2:
        """Add steps to the plan.

        Step can be provided as a sequence or as a plan. If provided as a plan, we will also take
        plan inputs from the plan, provided there are no duplicates (if there are duplicates, a
        PlanBuilderError will be raised).
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
        """Set the final output of the plan.

        Args:
            output_schema: The schema for the final output. If provided, an LLM will be used to
              coerce the output to this schema.
            summarize: Whether to summarize the final output. If True, a summary of the final output
              will be provided along with the value.

        """
        self.plan.final_output_schema = output_schema
        self.plan.summarize = summarize
        return self

    def build(self) -> PlanV2:
        """Return the plan, ready to run."""
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
