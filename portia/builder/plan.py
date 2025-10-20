"""Data model for plans assembled with :class:`PlanBuilder`."""

from __future__ import annotations

import uuid
from typing import Self

from pydantic import BaseModel, Field, model_validator

from portia.builder.reference import default_step_name
from portia.builder.step import Step
from portia.logger import logger
from portia.plan import PlanInput
from portia.plan import Step as StepData
from portia.prefixed_uuid import PlanUUID


class Plan(BaseModel):
    """An ordered collection of executable steps that can be executed by Portia.

    A Plan defines a sequence of Step objects that are executed in order to accomplish a
    specific task. Plans can include inputs, conditional logic, tool invocations, agent calls
    and structured outputs.
    """

    id: PlanUUID = Field(
        default_factory=PlanUUID,
        description="Unique identifier for the plan, automatically generated if not provided.",
    )
    steps: list[Step] = Field(
        description=(
            "Ordered sequence of steps to be executed. Each step is a Step instance representing "
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

    def to_step_data_list(self) -> list[StepData]:
        """Convert all steps to StepData objects for UI serialization.

        This method is used to convert the plan's steps into a format that can be
        displayed in the Portia Dashboard.

        """
        return [step.to_step_data(self) for step in self.steps]

    def step_output_name(self, step: int | str | Step) -> str:
        """Generate the output variable name for a given step.

        Creates a standardized variable name that can be used to reference the output
        of a specific step. If the step cannot be resolved, returns a placeholder
        name.

        Args:
            step: The step to get the output name for. Can be:
                - int: Index of the step in the plan (negative values count from the end)
                - str: Name of the step
                - Step: The step instance itself

        """
        try:
            if isinstance(step, Step):
                step_num = self.steps.index(step)
            elif isinstance(step, str):
                step_num = self.idx_by_name(step)
            else:
                step_num = step if step >= 0 else len(self.steps) + step
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
        step_data_list = []
        for step in self.steps:
            step_data = step.to_step_data(self)
            if step_data.tool_id:
                tools.append(step_data.tool_id)
            step_data_list.append(step_data)

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

        steps_section = "Steps:\n" + "\n".join([step.pretty_print() for step in step_data_list])

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
