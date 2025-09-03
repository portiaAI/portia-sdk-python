"""Data model for plans assembled with :class:`PlanBuilderV2`."""

from __future__ import annotations

import uuid
from typing import Self

from pydantic import BaseModel, Field, model_validator

from portia.builder.reference import default_step_name
from portia.builder.step_v2 import StepV2
from portia.logger import logger
from portia.plan import Plan, PlanContext, PlanInput
from portia.prefixed_uuid import PlanUUID


class PlanV2(BaseModel):
    """An ordered collection of executable steps that can be executed by Portia.

    A PlanV2 defines a sequence of StepV2 objects that are executed in order to accomplish a
    specific task. Plans can include inputs, conditional logic, tool invocations, agent calls
    and structured outputs.

    This class is the successor to the Plan class in Portia. You should use this class rather than
    the Plan class.
    """

    id: PlanUUID = Field(
        default_factory=PlanUUID,
        description="Unique identifier for the plan, automatically generated if not provided.",
    )
    steps: list[StepV2] = Field(
        description=(
            "Ordered sequence of steps to be executed. Each step is a StepV2 instance representing "
            "a specific action, tool invocation, agent call or control flow element."
        )
    )
    plan_inputs: list[PlanInput] = Field(
        default_factory=list,
        description=(
            "Input for the plan. These are values that are provided when the plan is "
            "executed, rather than when the plan is built. Steps in the plan can reference "
            "their values using the Input reference (e.g. args={'value': Input('input_name')}) "
            "and these are then resolved to the input value when the plan is executed."
        ),
    )
    summarize: bool = Field(
        default=False,
        description=(
            "Whether to generate a summary of the plan execution results. When True, "
            "Portia will create a concise summary of the key outputs and outcomes after "
            "all steps have completed."
        ),
    )
    final_output_schema: type[BaseModel] | None = Field(
        default=None,
        description=(
            "Optional Pydantic model schema defining the expected structure of the plan's "
            "final output. When provided, the plan execution results will be structured "
            "to match this schema, enabling type-safe consumption of plan outputs."
        ),
    )
    label: str = Field(
        default="Run the plan built with the Plan Builder",
        description=(
            "Human-readable description of the task or goal that this plan accomplishes. "
            "This label is used for display purposes in the Portia dashboard and helps "
            "users identify the plan's purpose."
        ),
    )

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        """Validate the plan structure and enforce uniqueness constraints.

        Ensures that all step names and plan input names are unique within the plan,
        preventing conflicts during execution and reference resolution.

        Raises:
            ValueError: If duplicate step names or plan input names are found.

        """
        # Check for duplicate step names
        step_names = [step.step_name for step in self.steps]
        if len(step_names) != len(set(step_names)):
            duplicates = [name for name in step_names if step_names.count(name) > 1]
            unique_duplicates = list(set(duplicates))
            raise ValueError(f"Duplicate step names found: {unique_duplicates}")

        # Check for duplicate plan input names
        input_names = [plan_input.name for plan_input in self.plan_inputs]
        if len(input_names) != len(set(input_names)):
            duplicates = [name for name in input_names if input_names.count(name) > 1]
            unique_duplicates = list(set(duplicates))
            raise ValueError(f"Duplicate plan input names found: {unique_duplicates}")

        return self

    def to_legacy_plan(self, plan_context: PlanContext) -> Plan:
        """Convert this plan to the legacy Plan format.

        This method enables backward compatibility with systems that still use the
        original plan representation. It transforms each StepV2 into its legacy
        equivalent while preserving all execution semantics.

        Args:
            plan_context: Context information including the original query and tool registry.

        """
        return Plan(
            id=self.id,
            plan_context=plan_context,
            steps=[step.to_legacy_step(self) for step in self.steps],
            plan_inputs=self.plan_inputs,
            structured_output_schema=self.final_output_schema,
        )

    def step_output_name(self, step: int | str | StepV2) -> str:
        """Generate the output variable name for a given step.

        Creates a standardized variable name that can be used to reference the output
        of a specific step. If the step cannot be resolved, returns a placeholder
        name.

        Args:
            step: The step to get the output name for. Can be:
                - int: Index of the step in the plan
                - str: Name of the step
                - StepV2: The step instance itself

        """
        try:
            if isinstance(step, StepV2):
                step_num = self.steps.index(step)
            elif isinstance(step, str):
                step_num = self.idx_by_name(step)
            else:
                step_num = step
        except ValueError:
            logger().warning(
                f"Attempted to retrieve name of step {step} but step not found in plan"
            )
            return f"$unknown_step_output_{uuid.uuid4().hex}"
        else:
            return f"${default_step_name(step_num)}_output"

    def idx_by_name(self, name: str) -> int:
        """Find the index of a step by its name.

        Searches through the plan's steps to find the one with the specified name
        and returns its position in the execution order.

        Raises:
            ValueError: If no step with the specified name exists in the plan.

        """
        for i, step in enumerate(self.steps):
            if step.step_name == name:
                return i
        raise ValueError(f"Step {name} not found in plan")

    def pretty_print(self) -> str:
        """Return a human-readable summary of the plan."""
        tools = []
        legacy_steps = []
        for step in self.steps:
            legacy_step = step.to_legacy_step(self)
            if legacy_step.tool_id:
                tools.append(legacy_step.tool_id)
            legacy_steps.append(legacy_step)

        unique_tools = sorted(set(tools))

        portia_tools = [tool for tool in unique_tools if tool.startswith("portia:")]
        other_tools = [tool for tool in unique_tools if not tool.startswith("portia:")]
        tools_summary = f"{len(portia_tools)} portia tools, {len(other_tools)} other tools"

        inputs_section = ""
        if self.plan_inputs:
            inputs_section = (
                "Inputs:\n    "
                + "\n    ".join([input_.pretty_print() for input_ in self.plan_inputs])
                + "\n"
            )

        steps_section = "Steps:\n" + "\n".join([step.pretty_print() for step in legacy_steps])

        final_output_section = ""
        if self.final_output_schema:
            final_output_section = f"\nFinal Output Schema: {self.final_output_schema.__name__}"

        return (
            f"Task: {self.label}\n"
            f"Tools Available Summary: {tools_summary}\n"
            f"{inputs_section}"
            f"{steps_section}\n"
            f"Summarize Plan Output: {self.summarize}"
            f"{final_output_section}"
        )
