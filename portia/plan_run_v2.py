"""PlanRunV2 for executing PlanV2 instances.

This module defines the new PlanRunV2 data structure that replaces the legacy PlanRun
for use with PlanV2. It includes the plan itself along with execution state.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from portia.builder.plan_v2 import PlanV2
from portia.config import Config
from portia.end_user import EndUser
from portia.execution_agents.output import LocalDataValue, Output
from portia.plan_run import PlanRunState
from portia.prefixed_uuid import PlanRunUUID
from portia.run_context import StepOutputValue


class PlanRunV2(BaseModel):
    """A plan run represents a running instance of a PlanV2.

    This is the V2 version that embeds the plan itself rather than just referencing it by ID.
    This simplifies the execution model by ensuring all necessary data is available in one place.

    Attributes:
        id (PlanRunUUID): A unique ID for this plan_run.
        state (PlanRunState): The current state of the PlanRun.
        current_step_index (int): The current step that is being executed.
        plan (PlanV2): The plan being executed.
        end_user (EndUser): The end user this plan was run for.
        step_output_values (list[StepOutputValue]): List of outputs from executed steps.
        final_output (Output): The final output of the plan run if available.
        plan_run_inputs (dict[str, LocalDataValue]): Dict mapping plan input names to their values.
        config (Config): The configuration for this plan run.

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
    plan: PlanV2 = Field(
        description="The PlanV2 being executed.",
    )
    end_user: EndUser = Field(
        description="The end user this plan was run for",
    )
    step_output_values: list[StepOutputValue] = Field(
        default_factory=list,
        description="List of outputs from executed steps.",
    )
    final_output: Output | None = Field(
        default=None,
        description="The final output of the plan run if available.",
    )
    plan_run_inputs: dict[str, LocalDataValue] = Field(
        default_factory=dict,
        description="Dict mapping plan input names to their values.",
    )
    config: Config = Field(
        description="The configuration for this plan run.",
    )

    def __str__(self) -> str:
        """Return the string representation of the PlanRunV2.

        Returns:
            str: A string representation containing key run attributes.

        """
        return (
            f"PlanRunV2(id={self.id}, plan_id={self.plan.id}, "
            f"state={self.state}, current_step_index={self.current_step_index}, "
            f"final_output={'set' if self.final_output else 'unset'})"
        )
