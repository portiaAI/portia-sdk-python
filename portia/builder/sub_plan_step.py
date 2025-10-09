"""Step that executes a sub-plan within a parent plan."""

from typing import Any, override

from langsmith import traceable
from pydantic import Field

from portia.builder.plan_v2 import PlanV2
from portia.builder.reference import Input
from portia.builder.step_v2 import StepV2
from portia.plan import PlanInput, Step
from portia.portia import Portia
from portia.run_context import RunContext


class SubPlanStep(StepV2):
    """A step that executes a nested PlanV2 and returns its final result.

    This step allows for modular plan composition by executing a complete sub-plan
    within the context of a larger plan. The sub-plan runs with its own input values
    and execution context, but shares the same tool registry, configuration, and
    execution hooks as the parent plan.

    This is useful for creating reusable sub-workflows, organizing complex plans
    into manageable components, or implementing conditional execution of entire
    plan segments. The sub-plan's final output becomes the output of this step,
    which can then be referenced by subsequent steps in the parent plan.
    """

    plan: PlanV2 = Field(description="The sub-plan to execute.")
    sub_portia: Portia | None = Field(
        default=None, exclude=True, description="Cached Portia instance for sub-plan execution."
    )

    def __str__(self) -> str:  # pragma: no cover - simple string representation
        """Return a description of this step for logging purposes."""
        return f"SubPlanStep(plan='{self.plan.label or self.plan.id}')"

    @override
    @traceable(name="Subplan Step - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Run the sub-plan using the current Portia configuration."""
        # Lazy initialization of Portia instance for reuse across multiple executions
        if self.sub_portia is None:
            self.sub_portia = Portia(
                config=run_data.config,
                tools=run_data.tool_registry,
                execution_hooks=run_data.execution_hooks,
                telemetry=run_data.telemetry,
            )

        plan_run_inputs: list[PlanInput] = []
        for plan_input in self.plan.plan_inputs:
            value = run_data.plan_run.plan_run_inputs.get(plan_input.name, plan_input.value)
            resolved_value = self._resolve_references(value, run_data)
            plan_run_inputs.append(PlanInput(name=plan_input.name, value=resolved_value))

        plan_run = await self.sub_portia.arun_plan(
            self.plan, run_data.end_user, plan_run_inputs=plan_run_inputs
        )

        if not plan_run.outputs.final_output:
            return None
        return plan_run.outputs.final_output.full_value(run_data.storage)

    @override
    def to_step_data(self, plan: PlanV2) -> Step:
        """Convert this SubPlanStep to a legacy Step."""
        tools = [s.to_step_data(plan).tool_id for s in self.plan.steps]
        inputs = [Input(plan_input.name) for plan_input in self.plan.plan_inputs]
        return Step(
            task=f"Run sub-plan: {self.plan.label}",
            inputs=self._inputs_to_legacy_plan_variables(inputs, plan),
            tool_id=",".join([t for t in tools if t is not None]),
            output=plan.step_output_name(self),
            condition=self._get_legacy_condition(plan),
        )
