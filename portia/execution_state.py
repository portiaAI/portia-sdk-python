"""Execution state dataclasses to replace heterogeneous tuples.

This module contains structured dataclasses that replace tuple passing in execution flow,
making the code more readable and type-safe. These dataclasses bundle related execution
state data that was previously passed as tuples.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from portia.end_user import EndUser
from portia.execution_agents.output import Output
from portia.introspection_agents.introspection_agent import PreStepIntrospection
from portia.plan import Plan
from portia.plan_run import PlanRun


class PlanRunReadinessResult(BaseModel):
    """Result of checking plan run initial readiness.

    Replaces tuple[bool, PlanRun] return from _check_initial_readiness.

    Attributes:
        is_ready: Whether the plan run is ready to execute
        plan_run: Updated plan run object (potentially modified with clarifications)

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_ready: bool = Field(
        description="Whether the plan run is ready to execute"
    )
    plan_run: PlanRun = Field(
        description="Updated plan run object (potentially modified with clarifications)"
    )


class IntrospectionResult(BaseModel):
    """Result of pre-step introspection processing.

    Replaces tuple[PlanRun, PreStepIntrospection] return from introspection methods.

    Attributes:
        plan_run: Updated plan run object after introspection processing
        introspection: Introspection result containing outcome and reason

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan_run: PlanRun = Field(
        description="Updated plan run object after introspection processing"
    )
    introspection: PreStepIntrospection = Field(
        description="Introspection result containing outcome (CONTINUE/SKIP/COMPLETE) and reason"
    )


class PlanRunSession(BaseModel):
    """Session data bundling Plan, PlanRun, EndUser, and last step output.

    This dataclass provides a structured way to pass around the core execution
    state that was previously passed as separate parameters or informal tuples.

    Attributes:
        plan: The plan being executed
        plan_run: Current plan run instance
        end_user: End user executing the plan
        last_step_output: Output from the last executed step, if any

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: Plan = Field(
        description="The plan being executed"
    )
    plan_run: PlanRun = Field(
        description="Current plan run instance"
    )
    end_user: EndUser = Field(
        description="End user executing the plan"
    )
    last_step_output: Output | None = Field(
        default=None,
        description="Output from the last executed step, if any"
    )

    @classmethod
    def create(
        cls,
        plan: Plan,
        plan_run: PlanRun,
        end_user: EndUser,
        last_step_output: Output | None = None,
    ) -> PlanRunSession:
        """Create a new PlanRunSession."""
        return cls(
            plan=plan,
            plan_run=plan_run,
            end_user=end_user,
            last_step_output=last_step_output,
        )


class PlanTestBundle(BaseModel):
    """Test utility bundling Plan and PlanRun.

    Replaces tuple[Plan, PlanRun] returns from test utility functions.

    Attributes:
        plan: The test plan
        plan_run: The test plan run

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: Plan = Field(
        description="The test plan"
    )
    plan_run: PlanRun = Field(
        description="The test plan run"
    )

    def unpack(self) -> tuple[Plan, PlanRun]:
        """Unpack to tuple for backwards compatibility during migration."""
        return self.plan, self.plan_run
