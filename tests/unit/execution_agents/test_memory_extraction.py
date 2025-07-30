"""Tests for memory extraction step."""

from __future__ import annotations

from unittest import mock

import pytest

from portia.end_user import EndUser
from portia.errors import InvalidPlanRunStateError
from portia.execution_agents.base_execution_agent import BaseExecutionAgent
from portia.execution_agents.memory_extraction import MemoryExtractionStep
from portia.execution_agents.output import LocalDataValue
from portia.plan import PlanBuilder, Variable
from portia.storage import InMemoryStorage
from tests.utils import get_test_config, get_test_plan_run


def test_memory_extraction_step_no_inputs() -> None:
    """Test MemoryExtractionStep with no step inputs."""
    (_, plan_run) = get_test_plan_run()
    agent = BaseExecutionAgent(
        plan=PlanBuilder().step(task="DESCRIPTION_STRING", output="$out").build(),
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
        plan=PlanBuilder()
        .step(
            task="DESCRIPTION_STRING",
            output="$out",
            inputs=[
                Variable(name="$local_output", description="Local input description"),
                Variable(name="$memory_output", description="Memory input description"),
            ],
        )
        .build(),
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
        plan=PlanBuilder()
        .step(
            task="DESCRIPTION_STRING",
            output="$out",
            inputs=[
                Variable(name="$missing_input", description="Missing input description"),
                Variable(name="$a", description="A value"),
            ],
        )
        .build(),
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
        "$plan_run_input": LocalDataValue(value="plan_run_input_value"),
    }

    agent = BaseExecutionAgent(
        plan=PlanBuilder()
        .step(
            task="DESCRIPTION_STRING",
            output="$out",
            inputs=[
                Variable(name="$plan_run_input", description="Plan run input description"),
            ],
        )
        .build(),
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


def test_memory_extraction_step_uses_summary_when_value_too_large() -> None:
    """Test MemoryExtractionStep uses summary when value exceeds context threshold."""
    (_, plan_run) = get_test_plan_run()

    plan_run.outputs.step_outputs = {
        "$large_output": LocalDataValue(
            value="x" * 10000, summary="This is a summary of the large value"
        ),
    }

    agent = BaseExecutionAgent(
        plan=PlanBuilder()
        .step(
            task="DESCRIPTION_STRING",
            output="$out",
            inputs=[
                Variable(name="$large_output", description="Large output description"),
            ],
        )
        .build(),
        plan_run=plan_run,
        config=get_test_config(),
        tool=None,
        agent_memory=InMemoryStorage(),
        end_user=EndUser(external_id="123"),
    )

    with mock.patch(
        "portia.execution_agents.memory_extraction.exceeds_context_threshold"
    ) as mock_threshold:
        mock_threshold.return_value = True
        memory_extraction_step = MemoryExtractionStep(agent=agent)
        result = memory_extraction_step.invoke({})

    assert result["step_inputs"][0].name == "$large_output"
    # The value should be the summary, not the original large value
    assert result["step_inputs"][0].value == "Large output description"
    assert result["step_inputs"][0].description == "Large output description"


def test_memory_extraction_step_uses_summaries_when_multiple_values_too_large() -> None:
    """Test MemoryExtractionStep handles multiple large inputs."""
    (_, plan_run) = get_test_plan_run()

    plan_run.outputs.step_outputs = {
        "$large_output_1": LocalDataValue(value="x" * 10000, summary="Summary of largest value"),
        "$large_output_2": LocalDataValue(
            value="y" * 8000, summary="Summary of second largest value"
        ),
        "$small_output": LocalDataValue(value="z", summary="Summary of small value"),
    }

    agent = BaseExecutionAgent(
        plan=PlanBuilder()
        .step(
            task="DESCRIPTION_STRING",
            output="$out",
            inputs=[
                Variable(name="$large_output_1", description="First large output description"),
                Variable(name="$large_output_2", description="Second large output description"),
                Variable(name="$small_output", description="Small output description"),
            ],
        )
        .build(),
        plan_run=plan_run,
        config=get_test_config(),
        tool=None,
        agent_memory=InMemoryStorage(),
        end_user=EndUser(external_id="123"),
    )

    with mock.patch(
        "portia.execution_agents.memory_extraction.exceeds_context_threshold"
    ) as mock_threshold:
        # Mock so that we exceed threshold on initial check + first 2 replacement checks, but pass
        # after replacing the 2 large values
        mock_threshold.side_effect = [True, True, True, False]
        memory_extraction_step = MemoryExtractionStep(agent=agent)
        result = memory_extraction_step.invoke({})

    assert result["step_inputs"][0].name == "$large_output_1"
    assert result["step_inputs"][0].value == "First large output description"
    assert result["step_inputs"][1].name == "$large_output_2"
    assert result["step_inputs"][1].value == "Second large output description"
    assert result["step_inputs"][2].name == "$small_output"
    assert result["step_inputs"][2].value == "z"
