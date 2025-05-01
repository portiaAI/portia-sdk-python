"""Tests for memory extraction step."""

from __future__ import annotations

import pytest

from portia.end_user import EndUser
from portia.errors import InvalidPlanRunStateError
from portia.execution_agents.base_execution_agent import BaseExecutionAgent
from portia.execution_agents.memory_extraction import MemoryExtractionStep
from portia.execution_agents.output import LocalDataValue
from portia.plan import Step, Variable
from portia.storage import InMemoryStorage
from tests.utils import get_test_config, get_test_plan_run


def test_memory_extraction_step_no_inputs() -> None:
    """Test MemoryExtractionStep with no step inputs."""
    (_, plan_run) = get_test_plan_run()
    agent = BaseExecutionAgent(
        step=Step(task="DESCRIPTION_STRING", output="$out"),
        plan_run=plan_run,
        config=get_test_config(),
        end_user=EndUser(external_id="123"),
        agent_memory=InMemoryStorage(),
        tool=None,
    )

    memory_extraction_step = MemoryExtractionStep(agent=agent)
    result = memory_extraction_step.invoke({})

    assert result == {"step_inputs": []}


def test_memory_extraction_step_with_inputs() -> None:
    """Test MemoryExtractionStep with step inputs (one local, one from agent memory)."""
    (_, plan_run) = get_test_plan_run()

    storage = InMemoryStorage()
    saved_output = storage.save_plan_run_output(
        "$memory_output",
        LocalDataValue(value="memory_value"),
        plan_run.id,
    )
    plan_run.outputs.step_outputs = {
        "$local_output": LocalDataValue(value="local_value"),
        "$memory_output": saved_output,
    }

    agent = BaseExecutionAgent(
        step=Step(
            task="DESCRIPTION_STRING",
            output="$out",
            inputs=[
                Variable(name="$local_output", description="Local input description"),
                Variable(name="$memory_output", description="Memory input description"),
            ],
        ),
        plan_run=plan_run,
        config=get_test_config(),
        tool=None,
        agent_memory=storage,
        end_user=EndUser(external_id="123"),
    )

    memory_extraction_step = MemoryExtractionStep(agent=agent)
    result = memory_extraction_step.invoke({})

    assert len(result["step_inputs"]) == 2
    assert result["step_inputs"][0].name == "$local_output"
    assert result["step_inputs"][0].value == "local_value"
    assert result["step_inputs"][0].description == "Local input description"
    assert result["step_inputs"][1].name == "$memory_output"
    assert result["step_inputs"][1].value == "memory_value"
    assert result["step_inputs"][1].description == "Memory input description"


def test_memory_extraction_step_errors_with_missing_input() -> None:
    """Test MemoryExtractionStep ignores step inputs that aren't in previous outputs."""
    (_, plan_run) = get_test_plan_run()
    agent = BaseExecutionAgent(
        step=Step(
            task="DESCRIPTION_STRING",
            output="$out",
            inputs=[
                Variable(name="$missing_input", description="Missing input description"),
                Variable(name="$a", description="A value"),
            ],
        ),
        plan_run=plan_run,
        config=get_test_config(),
        tool=None,
        agent_memory=InMemoryStorage(),
        end_user=EndUser(external_id="123"),
    )

    memory_extraction_step = MemoryExtractionStep(agent=agent)
    with pytest.raises(InvalidPlanRunStateError):
        memory_extraction_step.invoke({})


def test_memory_extraction_step_with_plan_run_inputs() -> None:
    """Test MemoryExtractionStep with inputs from plan_run_inputs."""
    (_, plan_run) = get_test_plan_run()
    plan_run.plan_run_inputs = {
        "$plan_run_input": "plan_run_input_value",
    }

    agent = BaseExecutionAgent(
        step=Step(
            task="DESCRIPTION_STRING",
            output="$out",
            inputs=[
                Variable(name="$plan_run_input", description="Plan run input description"),
            ],
        ),
        plan_run=plan_run,
        config=get_test_config(),
        tool=None,
        agent_memory=InMemoryStorage(),
        end_user=EndUser(external_id="123"),
    )

    memory_extraction_step = MemoryExtractionStep(agent=agent)
    result = memory_extraction_step.invoke({"messages": [], "step_inputs": []})

    assert len(result["step_inputs"]) == 1
    assert result["step_inputs"][0].name == "$plan_run_input"
    assert result["step_inputs"][0].value == "plan_run_input_value"
    assert result["step_inputs"][0].description == "Plan run input description"
