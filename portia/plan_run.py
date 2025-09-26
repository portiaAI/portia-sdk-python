"""Plan runs are executing instances of a Plan.

A plan run encapsulates all execution state, serving as the definitive record of its progress.
As the run runs, its `PlanRunState`, `current_step_index`, and `outputs` evolve to reflect
the current execution state.

The run also retains an `ExecutionContext`, which provides valuable insights for debugging
and analytics, capturing contextual information relevant to the run's execution.

Key Components
--------------
- **RunState**: Tracks the current status of the run (e.g., NOT_STARTED, IN_PROGRESS).
- **current_step_index**: Represents the step within the plan currently being executed.
- **outputs**: Stores the intermediate and final results of the PlanRun.
- **ExecutionContext**: Provides contextual metadata useful for logging and performance analysis.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from portia.clarification import (
    Clarification,
    ClarificationCategory,
    ClarificationListType,
)
from portia.common import PortiaEnum
from portia.execution_agents.output import LocalDataValue, Output
from portia.prefixed_uuid import PlanRunUUID, PlanUUID

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2
    from portia.config import Config
    from portia.end_user import EndUser


class PlanRunState(PortiaEnum):
    """The current state of the Plan Run.

    Attributes:
        NOT_STARTED: The run has not been started yet.
        IN_PROGRESS: The run is currently in progress.
        NEED_CLARIFICATION: The run requires further clarification before proceeding.
        READY_TO_RESUME: The run is ready to resume after clarifications have been resolved.
        COMPLETE: The run has been successfully completed.
        FAILED: The run has encountered an error and failed.

    """

    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    NEED_CLARIFICATION = "NEED_CLARIFICATION"
    READY_TO_RESUME = "READY_TO_RESUME"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class PlanRunOutputs(BaseModel):
    """Outputs of a Plan Run including clarifications.

    Attributes:
        clarifications (ClarificationListType): Clarifications raised by this plan run.
        step_outputs (dict[str, Output]): A dictionary containing outputs of individual steps.
            Outputs are indexed by the value given by the `step.output` field of the plan.
        final_output (Output | None): The final consolidated output of the PlanRun if available.

    """

    model_config = ConfigDict(extra="forbid")

    clarifications: ClarificationListType = Field(
        default=[],
        description="Clarifications raised by this plan_run.",
    )

    step_outputs: dict[str, Output] = Field(
        default={},
        description="A dictionary containing outputs of individual run steps.",
    )

    final_output: Output | None = Field(
        default=None,
        description="The final consolidated output of the PlanRun if available.",
    )


class PlanRun(BaseModel):
    """A plan run represents a running instance of a Plan.

    Attributes:
        id (PlanRunUUID): A unique ID for this plan_run.
        plan_id (PlanUUID): The ID of the Plan this run uses.
        current_step_index (int): The current step that is being executed.
        state (PlanRunState): The current state of the PlanRun.
        outputs (PlanRunOutputs): Outputs of the PlanRun including clarifications.
        plan_run_inputs (dict[str, LocalDataValue]): Dict mapping plan input names to their values.

    """

    model_config = ConfigDict(extra="forbid")

    id: PlanRunUUID = Field(
        default_factory=PlanRunUUID,
        description="A unique ID for this plan_run.",
    )
    plan_id: PlanUUID = Field(
        description="The ID of the Plan this run uses.",
    )
    current_step_index: int = Field(
        default=0,
        description="The current step that is being executed",
    )
    state: PlanRunState = Field(
        default=PlanRunState.NOT_STARTED,
        description="The current state of the PlanRun.",
    )
    end_user_id: str = Field(
        ...,
        description="The id of the end user this plan was run for",
    )
    outputs: PlanRunOutputs = Field(
        default=PlanRunOutputs(),
        description="Outputs of the run including clarifications.",
    )
    plan_run_inputs: dict[str, LocalDataValue] = Field(
        default_factory=dict,
        description="Dict mapping plan input names to their values.",
    )

    structured_output_schema: type[BaseModel] | None = Field(
        default=None,
        exclude=True,
        description="The optional structured output schema for the plan run.",
    )

    def get_outstanding_clarifications(self) -> ClarificationListType:
        """Return all outstanding clarifications.

        Returns:
            ClarificationListType: A list of outstanding clarifications that have not been resolved.

        """
        return [
            clarification
            for clarification in self.outputs.clarifications
            if not clarification.resolved
        ]

    def get_clarifications_for_step(self, step: int | None = None) -> ClarificationListType:
        """Return clarifications for the given step.

        Args:
            step (int | None): the step to get clarifications for. Defaults to current step.

        Returns:
            ClarificationListType: A list of clarifications for the given step.

        """
        if step is None:
            step = self.current_step_index
        return [
            clarification
            for clarification in self.outputs.clarifications
            if clarification.step == step
        ]

    def get_clarification_for_step(
        self, category: ClarificationCategory, step: int | None = None
    ) -> Clarification | None:
        """Return a clarification of the given category for the given step if it exists.

        Args:
            step (int | None): the step to get a clarification for. Defaults to current step.
            category (ClarificationCategory | None): the category of the clarification to get.

        """
        if step is None:
            step = self.current_step_index
        return next(
            (
                clarification
                for clarification in self.outputs.clarifications
                if clarification.step == step and clarification.category == category
            ),
            None,
        )

    def get_potential_step_inputs(self) -> dict[str, Output]:
        """Return a dictionary of potential step inputs for future steps."""
        return self.outputs.step_outputs | self.plan_run_inputs

    def __str__(self) -> str:
        """Return the string representation of the PlanRun.

        Returns:
            str: A string representation containing key run attributes.

        """
        return (
            f"Run(id={self.id}, plan_id={self.plan_id}, "
            f"state={self.state}, current_step_index={self.current_step_index}, "
            f"final_output={'set' if self.outputs.final_output else 'unset'})"
        )


class PlanRunV2(BaseModel):
    """Consolidated plan run data structure for the V2 migration.

    This class consolidates plan execution data into a single structure,
    containing all fields needed for plan execution and management.

    Attributes:
        id (PlanRunUUID): A unique ID for this plan run.
        state (PlanRunState): The current state of the PlanRun.
        current_step_index (int): The current step that is being executed.
        plan (PlanV2 | None): The PlanV2 instance being executed (V2 only).
        end_user (EndUser): The end user executing the plan.
        step_output_values (list[LocalDataValue]): Step output values from execution.
        final_output (Output | None): The final consolidated output.
        plan_run_inputs (dict[str, LocalDataValue]): Plan input values.
        config (Config): The Portia configuration.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: PlanRunUUID = Field(
        default_factory=PlanRunUUID,
        description="A unique ID for this plan run.",
    )
    state: PlanRunState = Field(
        default=PlanRunState.NOT_STARTED,
        description="The current state of the PlanRun.",
    )
    current_step_index: int = Field(
        default=0,
        description="The current step that is being executed",
    )
    plan: "PlanV2 | None" = Field(
        default=None,
        description="The PlanV2 instance being executed (V2 only).",
    )
    end_user: "EndUser" = Field(
        description="The end user executing the plan."
    )
    step_output_values: list["LocalDataValue"] = Field(
        default_factory=list,
        description="Step output values from execution.",
    )
    final_output: Output | None = Field(
        default=None,
        description="The final consolidated output of the PlanRun if available.",
    )
    plan_run_inputs: dict[str, "LocalDataValue"] = Field(
        default_factory=dict,
        description="Dict mapping plan input names to their values.",
    )
    config: "Config" = Field(
        description="The Portia configuration."
    )

    # Legacy compatibility fields for migration
    _legacy_outputs: PlanRunOutputs = Field(
        default=PlanRunOutputs(),
        description="Legacy outputs structure for compatibility.",
        exclude=True,
    )

    def get_outstanding_clarifications(self) -> ClarificationListType:
        """Return all outstanding clarifications.

        Returns:
            ClarificationListType: A list of outstanding clarifications that have not been resolved.
        """
        return [
            clarification
            for clarification in self._legacy_outputs.clarifications
            if not clarification.resolved
        ]

    def get_clarifications_for_step(self, step: int | None = None) -> ClarificationListType:
        """Return clarifications for the given step.

        Args:
            step (int | None): the step to get clarifications for. Defaults to current step.

        Returns:
            ClarificationListType: A list of clarifications for the given step.
        """
        if step is None:
            step = self.current_step_index
        return [
            clarification
            for clarification in self._legacy_outputs.clarifications
            if clarification.step == step
        ]

    def get_clarification_for_step(
        self, category: ClarificationCategory, step: int | None = None
    ) -> Clarification | None:
        """Return a clarification of the given category for the given step if it exists.

        Args:
            step (int | None): the step to get a clarification for. Defaults to current step.
            category (ClarificationCategory | None): the category of the clarification to get.
        """
        if step is None:
            step = self.current_step_index
        return next(
            (
                clarification
                for clarification in self._legacy_outputs.clarifications
                if clarification.step == step and clarification.category == category
            ),
            None,
        )

    def get_potential_step_inputs(self) -> dict[str, Output]:
        """Return a dictionary of potential step inputs for future steps."""
        return self._legacy_outputs.step_outputs | self.plan_run_inputs

    def __str__(self) -> str:
        """Return the string representation of the PlanRunV2.

        Returns:
            str: A string representation containing key run attributes.
        """
        plan_id = self.plan.id if self.plan else "unknown"
        return (
            f"PlanRunV2(id={self.id}, plan_id={plan_id}, "
            f"state={self.state}, current_step_index={self.current_step_index}, "
            f"final_output={'set' if self.final_output else 'unset'})"
        )


class ReadOnlyPlanRun(PlanRun):
    """A read-only copy of a Plan Run passed to agents for reference.

    This class provides a non-modifiable view of a plan run instance,
    ensuring that agents can access run details without altering them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def from_plan_run(cls, plan_run: PlanRun) -> ReadOnlyPlanRun:
        """Create a read-only plan run from a normal PlanRun.

        Args:
            plan_run (PlanRun): The original run instance to create a read-only copy from.

        Returns:
            ReadOnlyPlanRun: A new read-only instance of the provided PlanRun.

        """
        return cls(
            id=plan_run.id,
            plan_id=plan_run.plan_id,
            current_step_index=plan_run.current_step_index,
            outputs=plan_run.outputs,
            state=plan_run.state,
            end_user_id=plan_run.end_user_id,
            plan_run_inputs=plan_run.plan_run_inputs,
            structured_output_schema=plan_run.structured_output_schema,
        )


# Migration helpers for transitioning between old and new structures
def migrate_plan_run_to_v2(
    legacy_plan_run: PlanRun,
    plan_v2: "PlanV2 | None",
    end_user: "EndUser",
    config: "Config"
) -> "PlanRunV2":
    """Convert a legacy PlanRun to PlanRunV2.

    Args:
        legacy_plan_run: The legacy PlanRun instance to convert.
        plan_v2: The PlanV2 instance if available.
        end_user: The EndUser instance.
        config: The Config instance.

    Returns:
        PlanRunV2: The converted PlanRunV2 instance.
    """
    # Convert legacy step outputs to step_output_values
    step_output_values = []
    for output in legacy_plan_run.outputs.step_outputs.values():
        if hasattr(output, 'value'):
            step_output_values.append(LocalDataValue(value=output.value))

    return PlanRunV2(
        id=legacy_plan_run.id,
        state=legacy_plan_run.state,
        current_step_index=legacy_plan_run.current_step_index,
        plan=plan_v2,
        end_user=end_user,
        step_output_values=step_output_values,
        final_output=legacy_plan_run.outputs.final_output,
        plan_run_inputs=legacy_plan_run.plan_run_inputs,
        config=config,
        _legacy_outputs=legacy_plan_run.outputs,
    )


def migrate_v2_to_plan_run(plan_run_v2: "PlanRunV2") -> PlanRun:
    """Convert a PlanRunV2 back to legacy PlanRun for compatibility.

    Args:
        plan_run_v2: The PlanRunV2 instance to convert.

    Returns:
        PlanRun: The converted legacy PlanRun instance.
    """
    plan_id = plan_run_v2.plan.id if plan_run_v2.plan else "unknown"

    return PlanRun(
        id=plan_run_v2.id,
        plan_id=plan_id,
        current_step_index=plan_run_v2.current_step_index,
        state=plan_run_v2.state,
        end_user_id=plan_run_v2.end_user.external_id,
        outputs=plan_run_v2._legacy_outputs,
        plan_run_inputs=plan_run_v2.plan_run_inputs,
    )
