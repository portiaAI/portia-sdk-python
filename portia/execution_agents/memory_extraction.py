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
        step_inputs = []
        previous_outputs = self.agent.plan_run.outputs.step_outputs
        plan_run_inputs = self.agent.plan_run.plan_run_inputs

        for input_variable in self.agent.step.inputs:
            input_value = None

            if input_variable.name in previous_outputs:
                previous_output = previous_outputs.get(input_variable.name)
                input_value = previous_output.full_value(self.agent.agent_memory)  # pyright: ignore[reportOptionalMemberAccess]
            elif input_variable.name in plan_run_inputs:
                input_value = plan_run_inputs.get(input_variable.name)
            else:
                raise InvalidPlanRunStateError(
                    f"Received unknown step input: {input_variable.name}"
                )

            step_inputs.append(
                StepInput(
                    name=input_variable.name,
                    value=input_value,
                    description=input_variable.description,
                )
            )

        return {"step_inputs": step_inputs}
