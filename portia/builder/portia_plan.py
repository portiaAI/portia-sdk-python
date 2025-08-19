"""A plan built using the PlanBuilder."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from portia.builder.step import Step
from portia.logger import logger
from portia.plan import Plan, PlanContext, PlanInput
from portia.prefixed_uuid import PlanUUID


class PortiaPlan(BaseModel):
    """A sequence of steps to be run by Portia."""

    id: PlanUUID = Field(default_factory=PlanUUID, description="The ID of the plan.")
    steps: list[Step]
    plan_inputs: list[PlanInput] = Field(
        default=[],
        description="The inputs required by the plan.",
    )
    summarize: bool = False
    final_output_schema: type[BaseModel] | None = None
    task: str = Field(
        default="Run the plan build with the Plan Builder",
        description="The task that the plan is completing.",
    )

    def to_legacy_plan(self, plan_context: PlanContext) -> Plan:
        """Convert the Portia plan to a legacy plan."""
        return Plan(
            id=self.id,
            plan_context=plan_context,
            steps=[step.to_portia_step(self) for step in self.steps],
            plan_inputs=self.plan_inputs,
            structured_output_schema=self.final_output_schema,
        )

    def step_output_name(self, step: int | str | Step) -> str:
        """Get the name of the output of a step in the plan."""
        try:
            if isinstance(step, Step):
                step = self.steps.index(step)
            elif isinstance(step, str):
                step = self.idx_by_name(step)
        except ValueError:
            logger().warning(
                f"Attempted to retrieve name of step {step} but step not found in plan"
            )
            return f"$unknown_step_output_{uuid.uuid4().hex}"
        else:
            return f"$step_{step}_output"

    def idx_by_name(self, name: str) -> int:
        """Get the index of a step by name."""
        for i, step in enumerate(self.steps):
            if step.name == name:
                return i
        raise ValueError(f"Step {name} not found in plan")
