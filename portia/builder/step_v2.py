"""Implementation of the various step types used in :class:`PlanV2`."""

from __future__ import annotations

import itertools
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from portia.builder.conditionals import (
    ConditionalBlock,
)
from portia.builder.loops import LoopBlock
from portia.builder.reference import Input, Reference, StepOutput
from portia.execution_agents.output import LocalDataValue
from portia.plan import PlanInput, Step, Variable

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2
    from portia.run_context import RunContext


class StepData(BaseModel):
    """Lightweight dataclass representing the serializable view of a step for visualization.

    This provides the essential step information needed by the UI without
    the complexity of the full step implementation.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Unique identifier for this step within the plan.")
    type: str = Field(description="The type of step (e.g., 'LLMStep', 'UserInputStep').")
    name: str = Field(description="Human-readable name/label for this step.")
    task: str | None = Field(
        default=None, description="The task description for this step, if applicable."
    )
    inputs: list[str] = Field(
        default_factory=list, description="List of input names/references used by this step."
    )
    outputs: list[str] = Field(
        default_factory=list, description="List of output names produced by this step."
    )
    tool_id: str | None = Field(
        default=None, description="The ID of the tool used by this step, if applicable."
    )
    condition: str | None = Field(
        default=None, description="The condition controlling when this step runs, if any."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata specific to this step type."
    )


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

    def to_step_data(self, plan: PlanV2) -> StepData:
        """Convert this step to a StepData instance for visualization.

        This provides a lightweight, serializable representation of the step
        that contains the essential information needed by the UI.

        Args:
            plan: The plan containing this step.

        Returns:
            StepData: A lightweight representation of this step.

        """
        # Convert to legacy step first to get standardized data
        legacy_step = self.to_legacy_step(plan)

        # Extract input names from references
        input_names = []
        if hasattr(self, "inputs") and self.inputs:  # type: ignore[attr-defined]
            for input_item in self.inputs:  # type: ignore[attr-defined]
                if isinstance(input_item, Reference):
                    name = self._resolve_input_names_for_printing(input_item, plan)
                    if isinstance(name, str):
                        input_names.append(name)
                else:
                    # For non-reference inputs, just convert to string
                    input_names.append(str(input_item))

        # Create metadata dict with step-specific information
        metadata: dict[str, Any] = {}

        # Add any additional step-specific fields to metadata
        if hasattr(self, "output_schema") and self.output_schema is not None:  # type: ignore[attr-defined]
            metadata["output_schema"] = self.output_schema.__name__  # type: ignore[attr-defined]
        if hasattr(self, "system_prompt") and self.system_prompt is not None:  # type: ignore[attr-defined]
            metadata["system_prompt"] = self.system_prompt  # type: ignore[attr-defined]
        if hasattr(self, "model") and self.model is not None:  # type: ignore[attr-defined]
            metadata["model"] = str(self.model)  # type: ignore[attr-defined]
        if hasattr(self, "options") and self.options is not None:  # type: ignore[attr-defined]
            metadata["options"] = self.options  # type: ignore[attr-defined]
        if hasattr(self, "message") and hasattr(self, "options"):
            # This is a UserInputStep
            metadata["input_type"] = "multiple_choice" if self.options else "text"  # type: ignore[attr-defined]

        return StepData(
            id=self.step_name,
            type=self.__class__.__name__,
            name=self.step_name,
            task=getattr(self, "task", None),
            inputs=input_names,
            outputs=[legacy_step.output],
            tool_id=legacy_step.tool_id,
            condition=legacy_step.condition,
            metadata=metadata,
        )

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
            try:
                ref_obj = self._parse_reference_expression(full_match)
                resolved = self._resolve_references(ref_obj, run_data)
                return str(resolved)
            except (ValueError, AttributeError, KeyError):  # pragma: no cover
                # If parsing fails, return the original match
                return full_match  # pragma: no cover

        return re.sub(pattern, replace_reference, value)

    def _parse_reference_expression(self, templated_str: str) -> Reference:
        """Parse the content inside StepOutput(...) or Input(...) to create the reference object.

        Args:
            ref_cls: The Reference class to instantiate (StepOutput or Input)
            templated_str: The content inside the parentheses

        Supports the following reference types:
        - StepOutput(step_name)
        - StepOutput(step_name, path='field.name')
        - Input(input_name)

        """
        match templated_str:
            case s if "StepOutput" in s:
                return StepOutput.from_str(templated_str)
            case s if "Input" in s:
                return Input.from_str(templated_str)
            case _:  # pragma: no cover
                # this shouldn't be reachable unless we've added a new Reference type and haven't
                # updated this method to handle it
                raise ValueError(f"Invalid reference format: {templated_str}")  # pragma: no cover

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
