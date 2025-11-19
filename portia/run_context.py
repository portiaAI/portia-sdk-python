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
    """Data that is returned from a step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: PlanV2 = Field(description="The Portia plan being executed.")
    legacy_plan: Plan = Field(description="The legacy plan representation.")
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
        return ToolRunContext(
            end_user=self.end_user,
            plan_run=self.plan_run,
            plan=self.legacy_plan,
            config=self.config,
            clarifications=self.plan_run.get_clarifications_for_step(),
        )


class RunContextV2(BaseModel):
    """Updated context for PlanV2 runs using consolidated data structures.

    This is the V2 version of RunContext that uses the new consolidated
    PlanRunV2 structure instead of separate plan/legacy_plan/plan_run fields.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan_run: PlanRunV2 = Field(description="The consolidated plan run instance.")
    storage: Storage = Field(description="The Portia storage.")
    tool_registry: ToolRegistry = Field(description="The Portia tool registry.")
    execution_hooks: ExecutionHooks = Field(description="The Portia execution hooks.")
    telemetry: BaseProductTelemetry = Field(description="The Portia telemetry service.")

    def get_tool_run_ctx(self) -> ToolRunContext:
        """Get the tool run context for V2."""
        # Create a legacy plan if needed for backwards compatibility
        legacy_plan = None
        if self.plan_run.plan:
            from portia.plan import PlanContext
            # Create a minimal plan context for legacy compatibility
            plan_context = PlanContext(query="", tool_registry=self.tool_registry)
            legacy_plan = self.plan_run.plan.to_legacy_plan(plan_context)

        # Create a legacy PlanRun for ToolRunContext compatibility
        legacy_plan_run = PlanRun(
            id=self.plan_run.id,
            plan_id=self.plan_run.plan.id if self.plan_run.plan else "unknown",
            current_step_index=self.plan_run.current_step_index,
            state=self.plan_run.state,
            end_user_id=self.plan_run.end_user.external_id,
            outputs=self.plan_run._legacy_outputs,
            plan_run_inputs=self.plan_run.plan_run_inputs,
        )

        return ToolRunContext(
            end_user=self.plan_run.end_user,
            plan_run=legacy_plan_run,
            plan=legacy_plan,
            config=self.plan_run.config,
            clarifications=self.plan_run.get_clarifications_for_step(),
        )


# Migration helpers for RunContext structures
def migrate_run_context_to_v2(
    legacy_context: RunContext,
) -> RunContextV2:
    """Convert a legacy RunContext to RunContextV2.

    Args:
        legacy_context: The legacy RunContext instance to convert.

    Returns:
        RunContextV2: The converted RunContextV2 instance.
    """
    from portia.plan_run import migrate_plan_run_to_v2

    # Convert RunContext to RunContextV2 by migrating the PlanRun
    plan_run_v2 = migrate_plan_run_to_v2(
        legacy_plan_run=legacy_context.plan_run,
        plan_v2=legacy_context.plan,
        end_user=legacy_context.end_user,
        config=legacy_context.config,
    )

    return RunContextV2(
        plan_run=plan_run_v2,
        storage=legacy_context.storage,
        tool_registry=legacy_context.tool_registry,
        execution_hooks=legacy_context.execution_hooks,
        telemetry=legacy_context.telemetry,
    )


def migrate_v2_to_run_context(
    context_v2: RunContextV2,
) -> RunContext:
    """Convert a RunContextV2 back to legacy RunContext for compatibility.

    Args:
        context_v2: The RunContextV2 instance to convert.

    Returns:
        RunContext: The converted legacy RunContext instance.
    """
    from portia.plan_run import migrate_v2_to_plan_run

    # Convert back to legacy structures
    legacy_plan_run = migrate_v2_to_plan_run(context_v2.plan_run)

    # Create legacy plan if available
    legacy_plan = None
    if context_v2.plan_run.plan:
        from portia.plan import PlanContext
        plan_context = PlanContext(query="", tool_registry=context_v2.tool_registry)
        legacy_plan = context_v2.plan_run.plan.to_legacy_plan(plan_context)

    return RunContext(
        plan=context_v2.plan_run.plan if context_v2.plan_run.plan else None,
        legacy_plan=legacy_plan,
        plan_run=legacy_plan_run,
        end_user=context_v2.plan_run.end_user,
        step_output_values=context_v2.plan_run.step_output_values,
        config=context_v2.plan_run.config,
        storage=context_v2.storage,
        tool_registry=context_v2.tool_registry,
        execution_hooks=context_v2.execution_hooks,
        telemetry=context_v2.telemetry,
    )
