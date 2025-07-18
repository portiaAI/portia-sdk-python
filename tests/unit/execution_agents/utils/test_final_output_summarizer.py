"""Tests for the SummarizerAgent."""

from unittest import mock

import pytest
from pydantic import BaseModel

from portia.config import Config, GenerativeModelsConfig
from portia.execution_agents.output import LocalDataValue
from portia.execution_agents.utils.final_output_summarizer import FinalOutputSummarizer
from portia.introspection_agents.introspection_agent import (
    COMPLETED_OUTPUT,
    SKIPPED_OUTPUT,
)
from portia.model import GenerativeModel, Message
from portia.plan import Step
from tests.utils import get_test_config, get_test_plan_run


@pytest.fixture
def mock_summarizer_model() -> mock.MagicMock:
    """Mock the summarizer model."""
    return mock.MagicMock(spec=GenerativeModel)


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
    (plan, plan_run) = get_test_plan_run()
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

    summarizer = FinalOutputSummarizer(config=summarizer_config)
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
    (plan, plan_run) = get_test_plan_run()
    plan.plan_context.query = "Empty query"
    plan.steps = []
    plan_run.outputs.step_outputs = {}

    mock_summarizer_model.get_response.return_value = Message(
        content="Empty summary",
        role="assistant",
    )

    summarizer = FinalOutputSummarizer(config=summarizer_config)

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
    (plan, plan_run) = get_test_plan_run()
    plan.plan_context.query = "Test query"

    mock_summarizer_model.get_response.return_value = Message(content="", role="assistant")

    summarizer = FinalOutputSummarizer(config=summarizer_config)
    output = summarizer.create_summary(plan=plan, plan_run=plan_run)

    # Verify None handling
    assert output is None


def test_build_tasks_and_outputs_context(
    summarizer_config: Config,
) -> None:
    """Test that the tasks and outputs context is built correctly."""
    (plan, plan_run) = get_test_plan_run()

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

    summarizer = FinalOutputSummarizer(config=summarizer_config)
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
    (plan, plan_run) = get_test_plan_run()

    # Empty plan and run
    plan.plan_context.query = "Empty query"
    plan.steps = []
    plan_run.outputs.step_outputs = {}

    summarizer = FinalOutputSummarizer(config=get_test_config())
    context = summarizer._build_tasks_and_outputs_context(
        plan=plan,
        plan_run=plan_run,
    )

    # Should still include query even if no steps/outputs
    assert context == ("Query: Empty query\n----------")


def test_build_tasks_and_outputs_context_partial_outputs() -> None:
    """Test that the context builder handles steps with missing outputs."""
    (plan, plan_run) = get_test_plan_run()

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

    summarizer = FinalOutputSummarizer(config=get_test_config())
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
    (plan, plan_run) = get_test_plan_run()

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

    summarizer = FinalOutputSummarizer(config=get_test_config())
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
    (plan, plan_run) = get_test_plan_run()
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

    summarizer = FinalOutputSummarizer(config=summarizer_config)
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
