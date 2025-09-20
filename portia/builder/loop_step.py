"""Types to support Loops."""

from collections.abc import Callable, Sequence
from typing import Any, Self, override

from langsmith import traceable
from pydantic import Field, model_validator

from portia.builder.loops import LoopStepResult, LoopStepType, LoopType
from portia.builder.reference import Reference
from portia.builder.step import Step as StepV2
from portia.execution_agents.conditional_evaluation_agent import ConditionalEvaluationAgent
from portia.plan import Plan as PlanV2
from portia.plan import Step
from portia.run_context import RunContext


class LoopStep(StepV2):
    """A step that represents a loop in a loop block.

    This step handles loop logic such as while, do-while, and for-each loops that
    control which subsequent steps should be executed based on runtime conditions.
    """

    condition: Callable[..., bool] | str | None = Field(
        description=(
            "The boolean predicate to check. If evaluated to true, the loop will continue."
        )
    )
    over: Reference | Sequence[Any] | None = Field(
        default=None, description="The reference to loop over."
    )
    loop_type: LoopType
    index: int = Field(default=0, description="The current index of the loop.")
    args: dict[str, Reference | Any] = Field(
        default_factory=dict, description="The args to check the condition with."
    )
    loop_step_type: LoopStepType

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        """Validate the loop."""
        if self.condition is None and self.over is None:
            raise ValueError("Condition and over cannot both be None")
        if self.condition is not None and self.loop_type == LoopType.FOR_EACH:
            raise ValueError("Condition cannot be set for for-each loop")
        if self.condition and self.over:
            raise ValueError("Condition and over cannot both be set")
        if self.over is not None and self.loop_type in (LoopType.WHILE, LoopType.DO_WHILE):
            raise ValueError("Over cannot be set for while or do-while loop")
        return self

    def _current_loop_variable(self, run_data: RunContext) -> Any | None:  # noqa: ANN401
        """Get the current loop variable if over is set."""
        if self.over is None:
            return None
        if isinstance(self.over, str):
            values = [self.over]
        elif isinstance(self.over, Sequence):
            values = self.over[self.index]
        else:
            values = self._resolve_references(self.over, run_data)
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, Sequence):
            raise TypeError("Loop variable is not indexable")
        try:
            return values[self.index]
        except IndexError:
            return None

    @override
    @traceable(name="Loop Step - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the loop step."""
        args = {k: self._resolve_references(v, run_data) for k, v in self.args.items()}
        match self.loop_step_type, self.loop_type:
            case (LoopStepType.END, LoopType.DO_WHILE) | (LoopStepType.START, LoopType.WHILE):
                return await self._handle_conditional_loop(run_data, args)
            case LoopStepType.START, LoopType.FOR_EACH:
                if self.over is None:
                    raise ValueError("Over is required for for-each loop")
                value = self._current_loop_variable(run_data)
                self.index += 1
                return LoopStepResult(loop_result=value is not None, value=value)
            case _:
                return LoopStepResult(loop_result=True, value=True)

    async def _handle_conditional_loop(
        self, run_data: RunContext, args: dict[str, Any]
    ) -> LoopStepResult:
        if self.condition is None:
            raise ValueError("Condition is required for loop step")
        if isinstance(self.condition, str):
            template_reference = self._resolve_references(self.condition, run_data)
            agent = ConditionalEvaluationAgent(run_data.config)
            conditional_result = await agent.execute(
                conditional=str(template_reference), arguments=args
            )
        else:
            conditional_result = self.condition(**args)
        return LoopStepResult(loop_result=conditional_result, value=conditional_result)

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this LoopStep to a PlanStep."""
        if isinstance(self.condition, str):
            cond_str = self.condition
        else:
            cond_str = (
                "If result of "
                + getattr(self.condition, "__name__", str(self.condition))
                + " is true"
            )
        return Step(
            task=f"Loop clause: {cond_str}",
            inputs=self._inputs_to_legacy_plan_variables(list(self.args.values()), plan),
            tool_id=None,
            output=plan.step_output_name(self),
            condition=self._get_legacy_condition(plan),
        )
