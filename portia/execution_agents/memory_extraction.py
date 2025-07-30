"""Memory extraction step for execution agents.

This module provides a step that extracts memory from previous outputs for use in execution agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from portia.errors import InvalidPlanRunStateError
from portia.execution_agents.context import StepInput
from portia.token_check import exceeds_context_threshold

if TYPE_CHECKING:
    from portia.execution_agents.base_execution_agent import BaseExecutionAgent


class MemoryExtractionStep:
    """A step that extracts memory from the context."""

    def __init__(
        self,
        agent: BaseExecutionAgent,
    ) -> None:
        """Initialize the memory extraction step.

        Args:
            agent (BaseExecutionAgent): The agent using the memory extraction step.

        """
        self.agent = agent

    def invoke(self, _: dict[str, Any]) -> dict[str, Any]:
        """Invoke the model with the given message state.

        Returns:
            dict[str, Any]: The LangGraph state update to step_inputs

        """
        potential_inputs = self.agent.plan_run.get_potential_step_inputs()
        step_inputs = [
            StepInput(
                name=input_variable.name,
                value=potential_inputs[input_variable.name].full_value(self.agent.agent_memory),
                description=input_variable.description,
            )
            for input_variable in self.agent.step.inputs
            if input_variable.name in potential_inputs
        ]
        if exceeds_context_threshold(step_inputs, self.agent.config.get_execution_model(), 0.9):
            self._truncate_inputs(step_inputs)

        if len(step_inputs) != len(self.agent.step.inputs):
            expected_inputs = {input_.name for input_ in self.agent.step.inputs}
            known_inputs = {input_.name for input_ in step_inputs}
            raise InvalidPlanRunStateError(
                f"Received unknown step input(s): {expected_inputs - known_inputs}"
            )
        return {"step_inputs": step_inputs}

    def _truncate_inputs(self, inputs: list[StepInput]) -> None:
        """Truncate the step inputs so they fit in the context window."""
        # Replace input values with their description one by one (largest to smallest) until the
        # inputs fit
        inputs.sort(key=lambda x: len(str(x.value)) if x.value is not None else 0, reverse=True)
        for input_ in inputs:
            if not exceeds_context_threshold(inputs, self.agent.config.get_execution_model(), 0.9):
                return
            input_.value = input_.description
