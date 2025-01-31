"""Tests for the SummarizerAgent."""

from unittest import mock

from portia.agents.base_agent import Output
from portia.agents.utils.final_output_summarizer import FinalOutputSummarizer
from portia.plan import Step
from tests.utils import get_test_config, get_test_workflow


def test_summarizer_agent_execute_sync() -> None:
    """Test that the summarizer agent correctly executes and returns a summary."""
    # Set up test data
    (plan, workflow) = get_test_workflow()
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

    workflow.outputs.step_outputs = {
        "$london_weather": Output(value="Sunny and warm"),
        "$activities": Output(value="Visit Hyde Park and have a picnic"),
    }

    # Mock LLM response
    expected_summary = "Weather is sunny and warm in London, visit to Hyde Park for a picnic"
    mock_llm = mock.MagicMock()
    mock_llm.invoke.return_value.content = expected_summary

    with mock.patch("portia.agents.utils.final_output_summarizer.LLMWrapper") as mock_wrapper:
        mock_wrapper.return_value.to_langchain.return_value = mock_llm

        summarizer = FinalOutputSummarizer(config=get_test_config())
        output = summarizer.create_summary(plan=plan, workflow=workflow)

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
        expected_prompt = FinalOutputSummarizer.SUMMARIZE_TASK + expected_context
        mock_llm.invoke.assert_called_once_with(expected_prompt)


def test_summarizer_agent_empty_workflow() -> None:
    """Test summarizer agent with empty workflow."""
    (plan, workflow) = get_test_workflow()
    plan.plan_context.query = "Empty query"
    plan.steps = []
    workflow.outputs.step_outputs = {}

    mock_llm = mock.MagicMock()
    mock_llm.invoke.return_value.content = "Empty summary"

    with mock.patch("portia.agents.utils.final_output_summarizer.LLMWrapper") as mock_wrapper:
        mock_wrapper.return_value.to_langchain.return_value = mock_llm

        summarizer = FinalOutputSummarizer(config=get_test_config())
        output = summarizer.create_summary(plan=plan, workflow=workflow)

        # Verify empty context case
        assert output == "Empty summary"
        expected_prompt = FinalOutputSummarizer.SUMMARIZE_TASK + (
            "Query: Empty query\n----------"
        )
        mock_llm.invoke.assert_called_once_with(expected_prompt)


def test_summarizer_agent_handles_none_response() -> None:
    """Test that the agent handles None response from LLM."""
    (plan, workflow) = get_test_workflow()
    plan.plan_context.query = "Test query"

    mock_llm = mock.MagicMock()
    mock_llm.invoke.return_value.content = None

    with mock.patch("portia.agents.utils.final_output_summarizer.LLMWrapper") as mock_wrapper:
        mock_wrapper.return_value.to_langchain.return_value = mock_llm

        summarizer = FinalOutputSummarizer(config=get_test_config())
        output = summarizer.create_summary(plan=plan, workflow=workflow)

        # Verify None handling
        assert output is None


def test_build_tasks_and_outputs_context() -> None:
    """Test that the tasks and outputs context is built correctly."""
    (plan, workflow) = get_test_workflow()

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

    workflow.outputs.step_outputs = {
        "$london_weather": Output(value="Sunny and warm"),
        "$activities": Output(value="Visit Hyde Park and have a picnic"),
    }

    summarizer = FinalOutputSummarizer(config=get_test_config())
    context = summarizer._build_tasks_and_outputs_context(  # noqa: SLF001
        plan=plan,
        workflow=workflow,
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
    (plan, workflow) = get_test_workflow()

    # Empty plan and workflow
    plan.plan_context.query = "Empty query"
    plan.steps = []
    workflow.outputs.step_outputs = {}

    summarizer = FinalOutputSummarizer(config=get_test_config())
    context = summarizer._build_tasks_and_outputs_context(  # noqa: SLF001
        plan=plan,
        workflow=workflow,
    )

    # Should still include query even if no steps/outputs
    assert context == ("Query: Empty query\n----------")


def test_build_tasks_and_outputs_context_partial_outputs() -> None:
    """Test that the context builder handles steps with missing outputs."""
    (plan, workflow) = get_test_workflow()

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
    workflow.outputs.step_outputs = {
        "$london_weather": Output(value="Sunny and warm"),
    }

    summarizer = FinalOutputSummarizer(config=get_test_config())
    context = summarizer._build_tasks_and_outputs_context(  # noqa: SLF001
        plan=plan,
        workflow=workflow,
    )

    # Verify only step with output is included, but query is always present
    assert context == (
        "Query: What's the weather in London?\n"
        "----------\n"
        "Task: Get weather in London\n"
        "Output: Sunny and warm\n"
        "----------"
    )
