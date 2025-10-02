"""Context for a PlanV2 run."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from portia.builder.plan_v2 import PlanV2
from portia.config import Config
from portia.end_user import EndUser
from portia.execution_hooks import ExecutionHooks
from portia.plan import Plan
from portia.plan_run import PlanRun, PlanRunV2
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
    """Data that is returned from a step.

    This context object provides access to the PlanRunV2 execution state along with
    all necessary services and configuration for plan execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan_run_v2: PlanRunV2 = Field(description="The V2-native plan run instance.")
    step_output_values: list[StepOutputValue] = Field(
        default_factory=list, description="Outputs set by the step."
    )
    storage: Storage = Field(description="The Portia storage.")
    tool_registry: ToolRegistry = Field(description="The Portia tool registry.")
    execution_hooks: ExecutionHooks = Field(description="The Portia execution hooks.")
    telemetry: BaseProductTelemetry = Field(description="The Portia telemetry service.")

    # Legacy fields for backward compatibility
    plan: PlanV2 | None = Field(default=None, description="The Portia plan being executed.")
    legacy_plan: Plan | None = Field(default=None, description="The legacy plan representation.")
    plan_run: PlanRun | None = Field(default=None, description="The current plan run instance.")
    end_user: EndUser | None = Field(default=None, description="The end user executing the plan.")
    config: Config | None = Field(default=None, description="The Portia config.")

    def get_tool_run_ctx(self) -> ToolRunContext:
        """Get the tool run context."""
        return ToolRunContext(
            plan_run_v2=self.plan_run_v2,
            clarifications=self.plan_run_v2.get_clarifications_for_step(),
        )
