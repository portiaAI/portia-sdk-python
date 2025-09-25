"""Context for a PlanV2 run."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from portia.builder.plan_v2 import PlanV2
from portia.core.context import PortiaContext
from portia.end_user import EndUser
from portia.plan import Plan
from portia.plan_run import PlanRun
from portia.tool import ToolRunContext


class StepOutputValue(BaseModel):
    """Value that can be referenced by name."""

    value: Any = Field(description="The referenced value.")
    description: str = Field(description="Description of the referenced value.", default="")
    step_name: str = Field(description="The name of the referenced value.")
    step_num: int = Field(description="The step number of the referenced value.")


class RunContext(BaseModel):
    """Data that is returned from a step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: PlanV2 = Field(description="The Portia plan being executed.")
    legacy_plan: Plan = Field(description="The legacy plan representation.")
    plan_run: PlanRun = Field(description="The current plan run instance.")
    end_user: EndUser = Field(description="The end user executing the plan.")
    step_output_values: list[StepOutputValue] = Field(
        default_factory=list, description="Outputs set by the step."
    )
    context: PortiaContext = Field(description="The immutable Portia context.")

    def get_tool_run_ctx(self) -> ToolRunContext:
        """Get the tool run context."""
        return ToolRunContext(
            end_user=self.end_user,
            plan_run=self.plan_run,
            plan=self.legacy_plan,
            config=self.context.config,
            clarifications=self.plan_run.get_clarifications_for_step(),
        )
