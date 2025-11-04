"""Implementation of Conditional Steps."""

from __future__ import annotations

import sys
from collections.abc import Callable  # noqa: TC003
from typing import TYPE_CHECKING, Any

from portia.builder.conditionals import (
    ConditionalBlock,
    ConditionalBlockClauseType,
    ConditionalStepResult,
)
from portia.builder.reference import Reference  # noqa: TC001
from portia.builder.step import Step

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override  # pragma: no cover

from langsmith import traceable
from pydantic import Field, field_validator

from portia.execution_agents.conditional_evaluation_agent import ConditionalEvaluationAgent
from portia.plan import Step as StepData

if TYPE_CHECKING:
    from portia.builder.plan import Plan
    from portia.run_context import RunContext


class ConditionalStep(Step):
    """A step that represents a conditional clause within a conditional execution block.

    This step handles conditional logic such as if, else-if, else, and end-if statements
    that control which subsequent steps should be executed based on runtime conditions.
    """

    condition: Callable[..., bool] | str = Field(
        description=(
            "The condition to evaluate for this clause. Can be a callable that returns a boolean, "
            "or a string expression that will be evaluated by an LLM. If true, subsequent "
            "steps in this clause will execute; if false, execution jumps to the next clause."
        )
    )
    args: dict[str, Reference | Any] = Field(
        default_factory=dict,
        description=(
            "Arguments to pass to the condition. Values can be references to step outputs "
            "(using StepOutput), plan inputs (using Input), or literal values."
        ),
    )
    clause_index_in_block: int = Field(
        description="The position of this clause within its conditional block (0-based index)."
    )
    block_clause_type: ConditionalBlockClauseType = Field(
        description="The type of conditional clause (IF, ELIF, ELSE, or ENDIF)."
    )

    @field_validator("conditional_block", mode="after")
    @classmethod
    def validate_conditional_block(cls, v: ConditionalBlock | None) -> ConditionalBlock:
        """Validate the conditional block."""
        if v is None:
            raise ValueError("Conditional block is required for ConditionSteps")
        return v

    @property
    def block(self) -> ConditionalBlock:
        """Get the conditional block for this step."""
        if not isinstance(self.conditional_block, ConditionalBlock):
            raise TypeError("Conditional block is not a ConditionalBlock")
        return self.conditional_block

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        return (
            f"ConditionalStep(condition='{self.condition}', "
            f"clause_type='{self.block_clause_type.value}' args={self.args})"
        )

    @override
    @traceable(name="Conditional Step - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Evaluate the condition and return a ConditionalStepResult."""
        args = {k: self._resolve_references(v, run_data) for k, v in self.args.items()}
        if isinstance(self.condition, str):
            condition_str = self._template_references(self.condition, run_data)
            agent = ConditionalEvaluationAgent(run_data.config)
            conditional_result = await agent.execute(conditional=condition_str, arguments=args)
        else:
            conditional_result = self.condition(**args)
        next_clause_step_index = (
            self.block.clause_step_indexes[self.clause_index_in_block + 1]
            if self.clause_index_in_block < len(self.block.clause_step_indexes) - 1
            else self.block.clause_step_indexes[self.clause_index_in_block]
        )
        return ConditionalStepResult(
            type=self.block_clause_type,
            conditional_result=conditional_result,
            next_clause_step_index=next_clause_step_index,
            end_condition_block_step_index=self.block.clause_step_indexes[-1],
        )

    @override
    def to_step_data(self, plan: Plan) -> StepData:
        """Convert this ConditionalStep to a legacy Step."""
        if isinstance(self.condition, str):
            cond_str = self.condition
        else:
            cond_str = (
                "If result of "
                + getattr(self.condition, "__name__", str(self.condition))
                + " is true"
            )
        return Step(
            task=f"Conditional clause: {cond_str}",
            inputs=self._inputs_to_legacy_plan_variables(list(self.args.values()), plan),
            tool_id=None,
            output=plan.step_output_name(self),
            condition=self._get_legacy_condition(plan),
        )
