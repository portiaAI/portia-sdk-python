"""Context for a PlanV2 run."""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from portia.execution_hooks import ExecutionHooks
from portia.storage import Storage
from portia.telemetry.telemetry_service import BaseProductTelemetry
from portia.tool_registry import ToolRegistry

if TYPE_CHECKING:
    from portia.plan_run_v2 import PlanRunV2
    from portia.tool import ToolRunContext


class StepOutputValue(BaseModel):
    """Value that can be referenced by name."""

    value: Any = Field(description="The referenced value.")
    description: str = Field(description="Description of the referenced value.", default="")
    step_name: str = Field(description="The name of the referenced value.")
    step_num: int = Field(description="The step number of the referenced value.")


class RunContext(BaseModel):
    """Simplified execution context for PlanV2 runs.

    This context wraps PlanRunV2 and provides access to the execution infrastructure
    (storage, tool registry, hooks, telemetry) needed during plan execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan_run: "PlanRunV2" = Field(description="The current plan run instance.")
    storage: Storage = Field(description="The Portia storage.")
    tool_registry: ToolRegistry = Field(description="The Portia tool registry.")
    execution_hooks: ExecutionHooks = Field(description="The Portia execution hooks.")
    telemetry: BaseProductTelemetry = Field(description="The Portia telemetry service.")

    def get_tool_run_ctx(self) -> "ToolRunContext":
        """Get the tool run context.

        This creates a ToolRunContext with both V2 and legacy plan references for compatibility.
        """
        from portia.plan import PlanContext  # noqa: PLC0415
        from portia.tool import ToolRunContext  # noqa: PLC0415

        # Build legacy plan context from PlanV2
        tool_ids = []
        for step in self.plan_run.plan.steps:
            legacy_step = step.to_step_data(self.plan_run.plan)
            if legacy_step.tool_id:
                tool_ids.append(legacy_step.tool_id)

        plan_context = PlanContext(query=self.plan_run.plan.label, tool_ids=sorted(set(tool_ids)))

        legacy_plan = self.plan_run.plan.to_legacy_plan(plan_context)

        # Get clarifications - we'll need to adapt this from the old structure
        # For now, return empty list as clarifications are not in PlanRunV2 yet
        clarifications = []

        return ToolRunContext(
            end_user=self.plan_run.end_user,
            plan_run_v2=self.plan_run,
            plan_v2=self.plan_run.plan,
            plan=legacy_plan,
            config=self.plan_run.config,
            clarifications=clarifications,
        )
