"""Context for a PlanV2 run."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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
    """Context for executing a PlanV2 run.

    This class consolidates all the runtime state and dependencies needed to execute
    a plan. It holds a PlanRunV2 instance which contains the plan, config, end user,
    and execution state, plus the infrastructure services (storage, tool_registry, etc).

    Attributes:
        plan_run_v2 (PlanRunV2): The plan run instance containing execution state.
        storage (Storage): The Portia storage backend.
        tool_registry (ToolRegistry): The Portia tool registry.
        execution_hooks (ExecutionHooks): The Portia execution hooks.
        telemetry (BaseProductTelemetry): The Portia telemetry service.
        legacy_plan (Plan | None): Legacy plan representation for backward compatibility.
        legacy_plan_run (PlanRun | None): Legacy plan run for backward compatibility.
        step_output_values (list[StepOutputValue]): Step outputs for backward compatibility.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan_run_v2: PlanRunV2 = Field(description="The PlanRunV2 instance being executed.")
    storage: Storage = Field(description="The Portia storage.")
    tool_registry: ToolRegistry = Field(description="The Portia tool registry.")
    execution_hooks: ExecutionHooks = Field(description="The Portia execution hooks.")
    telemetry: BaseProductTelemetry = Field(description="The Portia telemetry service.")

    # Backward compatibility fields
    legacy_plan: Plan | None = Field(
        default=None, description="The legacy plan representation for backward compatibility."
    )
    legacy_plan_run: PlanRun | None = Field(
        default=None, description="The legacy plan run for backward compatibility."
    )
    step_output_values: list[StepOutputValue] = Field(
        default_factory=list, description="Outputs set by the step (for backward compatibility)."
    )

    def get_tool_run_ctx(self) -> ToolRunContext:
        """Get the tool run context.

        For backward compatibility, this uses the legacy_plan and legacy_plan_run if available,
        otherwise creates a temporary PlanRun from PlanRunV2.

        """
        if self.legacy_plan_run and self.legacy_plan:
            return ToolRunContext(
                end_user=self.plan_run_v2.end_user,
                plan_run=self.legacy_plan_run,
                plan=self.legacy_plan,
                config=self.plan_run_v2.config,
                clarifications=self.legacy_plan_run.get_clarifications_for_step(),
            )
        # Fallback for when legacy fields are not available
        # Create a minimal PlanRun for compatibility
        temp_plan_run = PlanRun(
            id=self.plan_run_v2.id,
            plan_id=self.plan_run_v2.plan.id,
            current_step_index=self.plan_run_v2.current_step_index,
            state=self.plan_run_v2.state,
            end_user_id=self.plan_run_v2.end_user.external_id,
            plan_run_inputs=self.plan_run_v2.plan_run_inputs,
        )
        # Create a minimal Plan for compatibility
        from portia.plan import PlanContext

        temp_plan_context = PlanContext(query="", tool_registry=self.tool_registry)
        temp_plan = self.plan_run_v2.plan.to_legacy_plan(temp_plan_context)

        return ToolRunContext(
            end_user=self.plan_run_v2.end_user,
            plan_run=temp_plan_run,
            plan=temp_plan,
            config=self.plan_run_v2.config,
            clarifications=[],
        )
