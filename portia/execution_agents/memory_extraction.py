"""Memory extraction step for execution agents.

This module provides a step that extracts memory from previous outputs for use in execution agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from portia.errors import InvalidPlanRunStateError
from portia.execution_agents.context import StepInput

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

        if len(step_inputs) != len(self.agent.step.inputs):
            expected_inputs = {input_.name for input_ in self.agent.step.inputs}
            known_inputs = {input_.name for input_ in step_inputs}
            raise InvalidPlanRunStateError(
                f"Received unknown step input(s): {expected_inputs - known_inputs}"
            )
        return {"step_inputs": step_inputs}
