"""Memory extraction step for execution agents.

This module provides a step that extracts memory from previous outputs for use in execution agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from portia.errors import InvalidPlanRunStateError
from portia.execution_agents.context import StepInput
from portia.execution_agents.output import AgentMemoryValue, LocalDataValue
from portia.logger import logger

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
        for step_input in self.agent.step.inputs:
            if step_input.name not in previous_outputs:
                raise InvalidPlanRunStateError("Received unknown step input: %s", step_input.name)
            previous_output = previous_outputs.get(step_input.name)

            match previous_output:
                case LocalDataValue():
                    output_value = previous_output.value
                case AgentMemoryValue():
                    output_value = self.agent.agent_memory.get_plan_run_output(
                        previous_output.output_name,
                        self.agent.plan_run.id,
                    ).value
                case _:
                    logger().warning(
                        "Received unknown output type: %s",
                        previous_output,
                    )
                    continue

            step_inputs.append(
                StepInput(
                    name=step_input.name,
                    value=output_value,
                    description=step_input.description,
                ),
            )
        return {"step_inputs": step_inputs}
