"""@@@ TODO"""

from __future__ import annotations

import contextlib

from pydantic import BaseModel, Field

from portia.builder.step import Step
from portia.plan import Plan, PlanContext
from portia.prefixed_uuid import PlanUUID


class PortiaPlan(BaseModel):
    """A sequence of steps to be run by Portia."""

    id: PlanUUID = Field(default_factory=PlanUUID, description="The ID of the plan.")
    steps: list[Step]

    def to_legacy_plan(self, plan_context: PlanContext) -> Plan:
        """Convert the Portia plan to a legacy plan."""
        return Plan(
            id=self.id,
            plan_context=plan_context,
            steps=[step.to_portia_step(self) for step in self.steps],
            plan_inputs=[],
        )

    def step_index(self, step: Step) -> int:
        """Get the index of a step in the plan."""
        with contextlib.suppress(ValueError):
            return self.steps.index(step)
        return -1
