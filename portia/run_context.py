"""Context for a PlanV2 run."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from portia.builder.plan_v2 import PlanV2
from portia.config import Config
from portia.end_user import EndUser
from portia.execution_agents.output import LocalDataValue, Output
from portia.execution_hooks import ExecutionHooks
from portia.plan import Plan
from portia.plan_run import PlanRunState
from portia.prefixed_uuid import PlanRunUUID
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


class PlanRunV2(BaseModel):
    """A plan run represents a running instance of a PlanV2.

    This consolidates all execution-specific state that was previously split between
    PlanRun and RunContext.

    Attributes:
        id (PlanRunUUID): A unique ID for this plan_run.
        state (PlanRunState): The current state of the PlanRun.
        current_step_index (int): The current step that is being executed.
        plan (PlanV2): The plan being executed.
        end_user (EndUser): The end user executing the plan.
        step_output_values (list[StepOutputValue]): Outputs set by the step.
        final_output (Output | None): The final consolidated output of the PlanRun.
        plan_run_inputs (dict[str, LocalDataValue]): Dict mapping plan input names to their values.
        config (Config): The Portia config.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: PlanRunUUID = Field(
        default_factory=PlanRunUUID,
        description="A unique ID for this plan_run.",
    )
    state: PlanRunState = Field(
        default=PlanRunState.NOT_STARTED,
        description="The current state of the PlanRun.",
    )
    current_step_index: int = Field(
        default=0,
        description="The current step that is being executed",
    )
    plan: PlanV2 = Field(description="The plan being executed.")
    end_user: EndUser = Field(description="The end user executing the plan.")
    step_output_values: list[StepOutputValue] = Field(
        default_factory=list, description="Outputs set by the step."
    )
    final_output: Output | None = Field(
        default=None,
        description="The final consolidated output of the PlanRun if available.",
    )
    plan_run_inputs: dict[str, LocalDataValue] = Field(
        default_factory=dict,
        description="Dict mapping plan input names to their values.",
    )
    config: Config = Field(description="The Portia config.")


class RunContext(BaseModel):
    """Context wrapper for a PlanV2 run.

    This class holds the PlanRunV2 instance along with environmental context
    like storage and tool registries.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan_run: PlanRunV2 = Field(description="The current plan run instance.")
    storage: Storage = Field(description="The Portia storage.")
    tool_registry: ToolRegistry = Field(description="The Portia tool registry.")
    execution_hooks: ExecutionHooks = Field(description="The Portia execution hooks.")
    telemetry: BaseProductTelemetry = Field(description="The Portia telemetry service.")

    # Deprecated fields kept for backwards compatibility
    @property
    def plan(self) -> PlanV2:
        """Get the plan from plan_run (backwards compatibility)."""
        return self.plan_run.plan

    @property
    def end_user(self) -> EndUser:
        """Get the end_user from plan_run (backwards compatibility)."""
        return self.plan_run.end_user

    @property
    def step_output_values(self) -> list[StepOutputValue]:
        """Get the step_output_values from plan_run (backwards compatibility)."""
        return self.plan_run.step_output_values

    @property
    def config(self) -> Config:
        """Get the config from plan_run (backwards compatibility)."""
        return self.plan_run.config

    @property
    def legacy_plan(self) -> Plan:
        """Get the legacy plan representation."""
        # We'll need to get this from storage or convert it
        # For now, create a placeholder - this will be properly implemented
        from portia.plan import PlanContext
        return self.plan_run.plan.to_legacy_plan(
            PlanContext(query="", tool_ids=[])
        )

    def get_tool_run_ctx(self) -> ToolRunContext:
        """Get the tool run context."""
        # Import here to avoid circular dependency
        from portia.plan_run import PlanRun

        # Create a legacy PlanRun for backwards compatibility with ToolRunContext
        legacy_plan_run = PlanRun(
            id=self.plan_run.id,
            plan_id=self.plan_run.plan.id,
            current_step_index=self.plan_run.current_step_index,
            state=self.plan_run.state,
            end_user_id=self.plan_run.end_user.external_id,
            plan_run_inputs=self.plan_run.plan_run_inputs,
        )

        return ToolRunContext(
            end_user=self.plan_run.end_user,
            plan_run=legacy_plan_run,
            plan=self.legacy_plan,
            config=self.plan_run.config,
            clarifications=[],  # Will be properly implemented
        )
