"""Tests for the SummarizerAgent."""

from unittest import mock

import pytest
from pydantic import BaseModel

from portia.config import Config, GenerativeModelsConfig
from portia.execution_agents.output import AgentMemoryValue, LocalDataValue
from portia.execution_agents.utils.final_output_summarizer import FinalOutputSummarizer
from portia.introspection_agents.introspection_agent import (
    COMPLETED_OUTPUT,
    SKIPPED_OUTPUT,
)
from portia.model import GenerativeModel, Message
from portia.plan import Step
from portia.storage import InMemoryStorage
from tests.utils import get_test_config, get_test_plan_run


@pytest.fixture
def mock_summarizer_model() -> mock.MagicMock:
    """Mock the summarizer model."""
    model = mock.MagicMock(spec=GenerativeModel)
    model.get_context_window_size.return_value = 100000
    return model


@pytest.fixture
def summarizer_config(mock_summarizer_model: mock.MagicMock) -> Config:
    """Create a summarizer config with a mocked model."""
    return get_test_config(models=GenerativeModelsConfig(summarizer_model=mock_summarizer_model))


def test_summarizer_agent_execute_sync(
    summarizer_config: Config,
    mock_summarizer_model: mock.MagicMock,
) -> None:
    """Test that the summarizer agent correctly executes and returns a summary."""
    # Set up test data
    test_bundle = get_test_plan_run()
    plan, plan_run = test_bundle.plan, test_bundle.plan_run
    plan.plan_context.query = "What's the weather in London and what can I do?"
    plan.steps = [
        Step(
            task="Get weather in London",
            output="$london_weather",
        ),
        Step(
            task="Suggest activities based on weather",
            output="$activities",
        ),
    ]

    plan_run.outputs.step_outputs = {
        "$london_weather": LocalDataValue(value="Sunny and warm"),
        "$activities": LocalDataValue(value="Visit Hyde Park and have a picnic"),
    }

    # Mock LLM response
    expected_summary = "Weather is sunny and warm in London, visit to Hyde Park for a picnic"
    mock_summarizer_model.get_response.return_value = Message(
        content=expected_summary,
        role="assistant",
    )

    summarizer = FinalOutputSummarizer(config=summarizer_config, agent_memory=InMemoryStorage())
    output = summarizer.create_summary(plan=plan, plan_run=plan_run)

    assert output == expected_summary

    # Verify LLM was called with correct prompt
    expected_context = (
        "Query: What's the weather in London and what can I do?\n"
        "----------\n"
        "Task: Get weather in London\n"
        "Output: Sunny and warm\n"
        "----------\n"
        "Task: Suggest activities based on weather\n"
        "Output: Visit Hyde Park and have a picnic\n"
        "----------"
    )
    expected_prompt = FinalOutputSummarizer.summarizer_only_prompt + expected_context
    mock_summarizer_model.get_response.assert_called_once_with(
        [Message(content=expected_prompt, role="user")],
    )


def test_summarizer_agent_empty_plan_run(
    summarizer_config: Config,
    mock_summarizer_model: mock.MagicMock,
) -> None:
    """Test summarizer agent with empty plan run."""
    test_bundle = get_test_plan_run()
    plan, plan_run = test_bundle.plan, test_bundle.plan_run
    plan.plan_context.query = "Empty query"
    plan.steps = []
    plan_run.outputs.step_outputs = {}

    mock_summarizer_model.get_response.return_value = Message(
        content="Empty summary",
        role="assistant",
    )

    summarizer = FinalOutputSummarizer(config=summarizer_config, agent_memory=InMemoryStorage())

    output = summarizer.create_summary(plan=plan, plan_run=plan_run)

    # Verify empty context case
    assert output == "Empty summary"
    expected_prompt = FinalOutputSummarizer.summarizer_only_prompt + (
        "Query: Empty query\n----------"
    )
    mock_summarizer_model.get_response.assert_called_once_with(
        [Message(content=expected_prompt, role="user")],
    )


def test_summarizer_agent_handles_empty_response(
    summarizer_config: Config,
    mock_summarizer_model: mock.MagicMock,
) -> None:
    """Test that the agent handles None response from LLM."""
    test_bundle = get_test_plan_run()
    plan, plan_run = test_bundle.plan, test_bundle.plan_run
    plan.plan_context.query = "Test query"

    mock_summarizer_model.get_response.return_value = Message(content="", role="assistant")

    summarizer = FinalOutputSummarizer(config=summarizer_config, agent_memory=InMemoryStorage())
    output = summarizer.create_summary(plan=plan, plan_run=plan_run)

    # Verify None handling
    assert output is None


def test_build_tasks_and_outputs_context(
    summarizer_config: Config,
) -> None:
    """Test that the tasks and outputs context is built correctly."""
    test_bundle = get_test_plan_run()
    plan, plan_run = test_bundle.plan, test_bundle.plan_run

    # Set up test data
    plan.plan_context.query = "What's the weather in London and what can I do?"
    plan.steps = [
        Step(
            task="Get weather in London",
            output="$london_weather",
        ),
        Step(
            task="Suggest activities based on weather",
            output="$activities",
        ),
    ]

    plan_run.outputs.step_outputs = {
        "$london_weather": LocalDataValue(value="Sunny and warm"),
        "$activities": LocalDataValue(value="Visit Hyde Park and have a picnic"),
    }

    summarizer = FinalOutputSummarizer(config=summarizer_config, agent_memory=InMemoryStorage())
    context = summarizer._build_tasks_and_outputs_context(
        plan=plan,
        plan_run=plan_run,
    )

    # Verify exact output format including query
    assert context == (
        "Query: What's the weather in London and what can I do?\n"
        "----------\n"
        "Task: Get weather in London\n"
        "Output: Sunny and warm\n"
        "----------\n"
        "Task: Suggest activities based on weather\n"
        "Output: Visit Hyde Park and have a picnic\n"
        "----------"
    )


def test_build_tasks_and_outputs_context_empty() -> None:
    """Test that the tasks and outputs context handles empty steps and outputs."""
    test_bundle = get_test_plan_run()
    plan, plan_run = test_bundle.plan, test_bundle.plan_run

    # Empty plan and run
    plan.plan_context.query = "Empty query"
    plan.steps = []
    plan_run.outputs.step_outputs = {}

    summarizer = FinalOutputSummarizer(config=get_test_config(), agent_memory=InMemoryStorage())
    context = summarizer._build_tasks_and_outputs_context(
        plan=plan,
        plan_run=plan_run,
    )

    # Should still include query even if no steps/outputs
    assert context == ("Query: Empty query\n----------")


def test_build_tasks_and_outputs_context_partial_outputs() -> None:
    """Test that the context builder handles steps with missing outputs."""
    test_bundle = get_test_plan_run()
    plan, plan_run = test_bundle.plan, test_bundle.plan_run

    # Set up test data with missing output
    plan.plan_context.query = "What's the weather in London?"
    plan.steps = [
        Step(
            task="Get weather in London",
            output="$london_weather",
        ),
        Step(
            task="Suggest activities based on weather",
            output="$activities",
        ),
    ]

    # Only provide output for first step
    plan_run.outputs.step_outputs = {
        "$london_weather": LocalDataValue(value="Sunny and warm"),
    }

    summarizer = FinalOutputSummarizer(config=get_test_config(), agent_memory=InMemoryStorage())
    context = summarizer._build_tasks_and_outputs_context(
        plan=plan,
        plan_run=plan_run,
    )

    # Verify only step with output is included, but query is always present
    assert context == (
        "Query: What's the weather in London?\n"
        "----------\n"
        "Task: Get weather in London\n"
        "Output: Sunny and warm\n"
        "----------"
    )


def test_build_tasks_and_outputs_context_with_conditional_outcomes() -> None:
    """Test that the context builder correctly uses summary for conditional outcomes."""
    test_bundle = get_test_plan_run()
    plan, plan_run = test_bundle.plan, test_bundle.plan_run

    plan.plan_context.query = "Test query with conditional outcomes"
    plan.steps = [
        Step(
            task="Regular task",
            output="$regular_output",
        ),
        Step(
            task="Skipped task",
            output="$skipped_output",
        ),
        Step(
            task="Complete task",
            output="$complete_output",
        ),
    ]

    plan_run.outputs.step_outputs = {
        "$regular_output": LocalDataValue(value="Regular result", summary="Not used"),
        "$skipped_output": LocalDataValue(
            value=SKIPPED_OUTPUT,
            summary="This task was skipped as it was unnecessary",
        ),
        "$complete_output": LocalDataValue(
            value=COMPLETED_OUTPUT,
            summary="The plan execution was completed early",
        ),
    }

    summarizer = FinalOutputSummarizer(config=get_test_config(), agent_memory=InMemoryStorage())
    context = summarizer._build_tasks_and_outputs_context(
        plan=plan,
        plan_run=plan_run,
    )

    assert context == (
        "Query: Test query with conditional outcomes\n"
        "----------\n"
        "Task: Regular task\n"
        "Output: Regular result\n"
        "----------\n"
        "Task: Skipped task\n"
        "Output: This task was skipped as it was unnecessary\n"
        "----------\n"
        "Task: Complete task\n"
        "Output: The plan execution was completed early\n"
        "----------"
    )


def test_summarizer_agent_handles_structured_output_with_fo_summary(
    summarizer_config: Config,
    mock_summarizer_model: mock.MagicMock,
) -> None:
    """Test that the summarizer agent correctly executes and returns a structured output."""
    # Set up test data
    test_bundle = get_test_plan_run()
    plan, plan_run = test_bundle.plan, test_bundle.plan_run
    plan.plan_context.query = "What's the weather in London and what can I do?"
    plan.steps = [
        Step(
            task="Get weather in London",
            output="$london_weather",
        ),
        Step(
            task="Suggest activities based on weather",
            output="$activities",
        ),
    ]

    plan_run.outputs.step_outputs = {
        "$london_weather": LocalDataValue(value="Sunny and warm"),
        "$activities": LocalDataValue(value="Visit Hyde Park and have a picnic"),
    }

    class TestStructuredOutput(BaseModel):
        mock_field: str

    plan_run.structured_output_schema = TestStructuredOutput

    # Create a mock response that matches the structure we expect
    class SchemaWithSummary(TestStructuredOutput):
        fo_summary: str

    mock_response = SchemaWithSummary(mock_field="mock_value", fo_summary="mock_summary")
    mock_summarizer_model.get_structured_response.return_value = mock_response

    summarizer = FinalOutputSummarizer(config=summarizer_config, agent_memory=InMemoryStorage())
    output = summarizer.create_summary(plan=plan, plan_run=plan_run)

    assert isinstance(output, SchemaWithSummary)
    assert output.mock_field == mock_response.mock_field
    assert output.fo_summary == mock_response.fo_summary

    # Verify LLM was called with correct prompt
    expected_context = (
        "Query: What's the weather in London and what can I do?\n"
        "----------\n"
        "Task: Get weather in London\n"
        "Output: Sunny and warm\n"
        "----------\n"
        "Task: Suggest activities based on weather\n"
        "Output: Visit Hyde Park and have a picnic\n"
        "----------"
    )
    expected_prompt = (
        FinalOutputSummarizer.summarizer_and_structured_output_prompt + expected_context
    )
    mock_summarizer_model.get_structured_response.assert_called_once_with(
        [Message(content=expected_prompt, role="user")],
        mock.ANY,  # Use mock.ANY since we can't predict the exact dynamic class
    )


def test_summarizer_agent_handles_large_response(
    summarizer_config: Config,
    mock_summarizer_model: mock.MagicMock,
) -> None:
    """Test that the agent handles large response from the tool correctly."""
    test_bundle = get_test_plan_run()
    plan, plan_run = test_bundle.plan, test_bundle.plan_run
    plan.steps = [
        Step(
            task="Process very large dataset",
            output="$large_output",
        ),
    ]
    large_memory_output = AgentMemoryValue(
        output_name="large_data",
        plan_run_id=plan_run.id,
        summary="Summary of large value",
    )
    plan_run.outputs.step_outputs = {
        "$large_output": large_memory_output,
    }

    mock_memory = mock.MagicMock()
    large_retrieved_output = LocalDataValue(
        value="A" * 50000,
        summary="Summary of large value",
    )
    mock_memory.get_plan_run_output.return_value = large_retrieved_output

    with mock.patch(
        "portia.execution_agents.utils.final_output_summarizer.exceeds_context_threshold"
    ) as mock_threshold:
        mock_threshold.return_value = True

        mock_summarizer_model.get_response.return_value = Message(
            content="Summary of large processing",
            role="assistant",
        )

        summarizer = FinalOutputSummarizer(config=summarizer_config, agent_memory=mock_memory)
        output = summarizer.create_summary(plan=plan, plan_run=plan_run)
        assert output == "Summary of large processing"

        mock_threshold.assert_called_once_with(
            large_retrieved_output.value, summarizer_config.get_summarizer_model(), 0.9
        )
        assert mock_summarizer_model.get_response.call_count == 1
        assert (
            "Summary of large value"
            in mock_summarizer_model.get_response.call_args[0][0][0].content
        )
        assert "AAAAA" not in mock_summarizer_model.get_response.call_args[0][0][0].content
