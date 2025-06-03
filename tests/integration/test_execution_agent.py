"""Tests for the execution agent."""

from typing import Any

import pytest
from pydantic import BaseModel

from portia.config import Config
from portia.end_user import EndUser
from portia.execution_agents.one_shot_agent import OneShotAgent
from portia.execution_agents.output import LocalDataValue
from portia.plan import PlanBuilder
from portia.plan_run import PlanRun
from portia.storage import InMemoryStorage
from portia.tool import Tool, ToolRunContext


class QuantumEntanglementSimulatorArgs(BaseModel):
    """Arguments for the Quantum Entanglement Simulator tool."""

    particle_count: int
    simulation_time_femtoseconds: float
    initial_state_vector: list[float]
    measurement_basis: str = "computational"
    environmental_noise_model: str = "gaussian"
    noise_amplitude: float = 0.01


class QuantumEntanglementSimulatorTool(Tool):
    """A tool with a very long description."""

    id: str = "quantum_entanglement_simulator"
    name: str = "Quantum Entanglement Simulator"
    description: str = """
    Simulates the quantum entanglement dynamics of a system of qubits over a specified time period.
    This tool models the evolution of the quantum state under the influence of both internal
    Hamiltonian dynamics and external environmental noise. It allows for the specification of
    various parameters to control the simulation environment and the initial conditions of the
    quantum system. The simulation calculates the time-dependent density matrix of the system, from
    which various entanglement measures (like concurrence or entanglement entropy) can be derived.
    Users can specify the number of particles (qubits), the total simulation time in femtoseconds,
    and the initial state vector representing the system's quantum state at t=0. The measurement
    basis for final state analysis can be chosen (e.g., 'computational', 'bell'). Furthermore,
    the tool supports different environmental noise models, such as 'gaussian' or 'decoherence',
    allowing investigation into the robustness of entanglement under realistic conditions. The
    amplitude of the noise can also be adjusted. The output includes the final density matrix,
    time series data of entanglement measures, and visualization plots if requested (plotting
    functionality might depend on available libraries). This tool is computationally intensive and
    recommended for systems with a small number of qubits (typically < 10) unless run on
    high-performance computing resources. Ensure the initial state vector is normalized and has the
    correct dimensions (2^particle_count). The simulation uses the Lindblad master equation for
    open quantum systems when noise models are active. Advanced users can potentially provide
    custom Hamiltonian functions or noise operators, though this requires specific formatting and
    validation. The simulation results are crucial for research in quantum information processing,
    quantum computing feasibility studies, and fundamental tests of quantum mechanics. It can help
    predict the lifetime of entangled states in different physical implementations of qubits,
    guiding experimental efforts. Parameter ranges: particle_count (2-10),
    simulation_time_femtoseconds (1.0-10000.0), noise_amplitude (0.0-0.5). Default measurement
    basis is 'computational', default noise model is 'gaussian'. Example Usage: Simulate a 3-qubit
    system initially in the GHZ state for 500 fs with 5% Gaussian noise. Provide
    initial_state_vector accordingly. Check documentation for state vector format details. The
    complexity scales exponentially with particle_count.
    """
    args_schema: type[BaseModel] = QuantumEntanglementSimulatorArgs
    output_schema: tuple[str, str] = (
        "json",
        "JSON containing final density matrix, entanglement measures time series, and optional "
        "metadata.",
    )

    def run(self, _: ToolRunContext, **__: Any) -> dict[str, Any]:
        """Run the tool."""
        return {"result": 42}


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet-latest",
        "google/gemini-2.0-flash",
    ],
)
def test_execution_agent_with_long_tool_description(model: str) -> None:
    """Test the execution agent with a tool with a long description."""
    config = Config.from_default(
        default_model=model,
    )
    tool = QuantumEntanglementSimulatorTool()
    end_user = EndUser(external_id="test_user")

    # Create Plan
    plan = (
        PlanBuilder("Simulate quantum entanglement")
        .step(
            task="Run the quantum simulator for a 3-qubit system (particle_count=3) for 500 "
            "femtoseconds (simulation_time_femtoseconds=500.0). Use an initial state vector "
            "representing the GHZ state [0.707, 0, 0, 0, 0, 0, 0, 0.707] (initial_state_vector"
            "=[0.707, 0, 0, 0, 0, 0, 0, 0.707]). Use the 'computational' measurement basis "
            "and 'gaussian' noise model with an amplitude of 0.05 (noise_amplitude=0.05).",
            tool_id=tool.id,
            output="$result",
        )
        .build()
    )

    # Create PlanRun
    plan_run = PlanRun(plan_id=plan.id, end_user_id=end_user.external_id)

    # Create AgentMemory
    agent_memory = InMemoryStorage()

    # Instantiate Agent
    agent = OneShotAgent(
        plan=plan,
        plan_run=plan_run,
        config=config,
        agent_memory=agent_memory,
        end_user=end_user,
        tool=tool,
    )

    # Execute Agent
    output = agent.execute_sync()

    # Assert Output
    assert isinstance(output, LocalDataValue)
    assert output.value == {"result": 42}
