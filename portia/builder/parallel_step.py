"""Implementation of the parallel step for concurrent task execution."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override  # pragma: no cover

from langsmith import traceable
from pydantic import Field

from portia.builder.step_v2 import StepV2
from portia.plan import Step

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2
    from portia.run_context import RunContext


class ParallelStep(StepV2):
    """A step that executes multiple child steps concurrently.

    This step allows developers to define a block of steps that should be executed
    in parallel rather than sequentially. This is useful for performance optimization
    when multiple independent tasks (e.g., calling different tools, running multiple
    LLM queries) can be performed simultaneously.

    The output of a ParallelStep is a list containing the outputs of its child steps,
    in the order they were provided. Subsequent steps can reference the outputs of
    individual child steps via indexing (e.g., StepOutput("parallel_step", path="0")).

    If any child step fails, all other concurrently running steps are cancelled,
    and the ParallelStep as a whole fails, propagating the error.
    """

    steps: list[StepV2] = Field(
        description="List of steps to execute in parallel. These steps should be independent "
        "with no data dependencies on one another."
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        step_types = [step.__class__.__name__ for step in self.steps]
        return f"ParallelStep(steps=[{', '.join(step_types)}])"

    @override
    @traceable(name="Parallel Step - Run")
    async def run(self, run_data: RunContext) -> list[Any]:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Execute all child steps concurrently and return their outputs as a list.

        Uses asyncio.gather to run all child steps in parallel. If any step fails,
        all other steps are cancelled and the exception is propagated.

        Args:
            run_data: The runtime context containing step outputs, inputs, and other
                execution data.

        Returns:
            A list of outputs from each child step, in the order they were provided.

        Raises:
            Exception: If any child step fails during execution.

        """
        # Create tasks for all child steps
        tasks = [step.run(run_data) for step in self.steps]

        # Execute all tasks concurrently
        # If any task fails, gather will cancel remaining tasks and raise the exception
        results = await asyncio.gather(*tasks)
        return list(results)

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this ParallelStep to a legacy Step.

        This creates a legacy step representation primarily for dashboard display.

        Args:
            plan: The PlanV2 instance containing this step.

        Returns:
            A legacy Step object representing this parallel step.

        """
        child_step_descriptions = [str(step) for step in self.steps]
        task_description = (
            f"Execute {len(self.steps)} steps in parallel: {', '.join(child_step_descriptions)}"
        )

        return Step(
            task=task_description,
            inputs=[],
            output=plan.step_output_name(self),
            condition=self._get_legacy_condition(plan),
        )
