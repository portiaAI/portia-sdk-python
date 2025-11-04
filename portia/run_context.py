"""Context for a Plan run."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from portia.builder.plan import Plan
from portia.config import Config
from portia.end_user import EndUser
from portia.execution_hooks import ExecutionHooks
from portia.plan import LegacyPlan, PlanContext
from portia.plan_run import PlanRun
from portia.storage import Storage
from portia.telemetry.telemetry_service import BaseProductTelemetry
from portia.tool import ToolRunContext
from portia.tool_registry import ToolRegistry


class StepOutputValue(BaseModel):
    """Value that can be referenced by name."""

    value: Any = Field(description="The referenced value.")
    description: str = Field(description="Description of the referenced value.", default="")
    step_name: str = Field(description="The name of the referenced value.")
    step_num: int = Field(description="The step number of the referenced value.")


class RunContext(BaseModel):
    """Data that is returned from a step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: Plan = Field(description="The Portia plan being executed.")
    plan_run: PlanRun = Field(description="The current plan run instance.")
    end_user: EndUser = Field(description="The end user executing the plan.")
    step_output_values: list[StepOutputValue] = Field(
        default_factory=list, description="Outputs set by the step."
    )
    config: Config = Field(description="The Portia config.")
    storage: Storage = Field(description="The Portia storage.")
    tool_registry: ToolRegistry = Field(description="The Portia tool registry.")
    execution_hooks: ExecutionHooks = Field(description="The Portia execution hooks.")
    telemetry: BaseProductTelemetry = Field(description="The Portia telemetry service.")

    def get_tool_run_ctx(self) -> ToolRunContext:
        """Get the tool run context."""
        # Create a LegacyPlan for backward compatibility
        step_data_list = self.plan.to_step_data_list()
        tool_ids = list({step.tool_id for step in step_data_list if step.tool_id})
        legacy_plan = LegacyPlan(
            id=self.plan.id,
            plan_context=PlanContext(query=self.plan.label, tool_ids=tool_ids),
            steps=step_data_list,
            plan_inputs=self.plan.plan_inputs,
            structured_output_schema=self.plan.final_output_schema,
        )
        return ToolRunContext(
            end_user=self.end_user,
            plan_run=self.plan_run,
            plan=legacy_plan,
            config=self.config,
            clarifications=self.plan_run.get_clarifications_for_step(),
        )
