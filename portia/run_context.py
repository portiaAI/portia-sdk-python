"""Context for a PlanV2 run."""

from pydantic import BaseModel, ConfigDict, Field

from portia.execution_hooks import ExecutionHooks
from portia.plan_run import PlanRunV2
from portia.storage import Storage
from portia.telemetry.telemetry_service import BaseProductTelemetry
from portia.tool import ToolRunContext
from portia.tool_registry import ToolRegistry


class RunContext(BaseModel):
    """Context for executing a PlanV2.

    This is the main context object used throughout PlanV2 execution.
    It contains the PlanRunV2 instance along with all services needed
    for execution (storage, tools, hooks, etc.).

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan_run: PlanRunV2 = Field(description="The current plan run instance.")
    storage: Storage = Field(description="The Portia storage.")
    tool_registry: ToolRegistry = Field(description="The Portia tool registry.")
    execution_hooks: ExecutionHooks = Field(description="The Portia execution hooks.")
    telemetry: BaseProductTelemetry = Field(description="The Portia telemetry service.")

    def get_tool_run_ctx(self) -> ToolRunContext:
        """Get the tool run context."""
        return ToolRunContext(
            plan_run=self.plan_run,
            clarifications=self.plan_run.get_clarifications_for_step(),
        )
