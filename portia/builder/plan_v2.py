"""A plan built using the PlanBuilder."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field, model_validator

from portia.builder.reference import default_step_name
from portia.builder.step_v2 import StepV2
from portia.logger import logger
from portia.plan import Plan, PlanContext, PlanInput
from portia.prefixed_uuid import PlanUUID


class PlanV2(BaseModel):
    """A sequence of steps to be run by Portia."""

    id: PlanUUID = Field(default_factory=PlanUUID, description="The ID of the plan.")
    steps: list[StepV2] = Field(description="The steps to be executed in the plan.")
    plan_inputs: list[PlanInput] = Field(
        default=[],
        description="The inputs required by the plan.",
    )
    summarize: bool = Field(default=False, description="Whether to summarize the plan output.")
    final_output_schema: type[BaseModel] | None = Field(
        default=None, description="The schema of the final output of the plan."
    )
    label: str = Field(
        default="Run the plan built with the Plan Builder",
        description="The task that the plan is completing.",
    )

    @model_validator(mode="after")
    def validate_plan(self) -> PlanV2:
        """Validate the plan."""
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
        """Convert the Portia plan to a legacy plan."""
        return Plan(
            id=self.id,
            plan_context=plan_context,
            steps=[step.to_legacy_step(self) for step in self.steps],
            plan_inputs=self.plan_inputs,
            structured_output_schema=self.final_output_schema,
        )

    def step_output_name(self, step: int | str | StepV2) -> str:
        """Get the name of the output of a step in the plan."""
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
        """Get the index of a step by name."""
        for i, step in enumerate(self.steps):
            if step.step_name == name:
                return i
        raise ValueError(f"Step {name} not found in plan")

    def pretty_print(self) -> str:
        """Return the pretty print representation of the plan.

        Returns:
            str: A pretty print representation of the plan.

        """
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
