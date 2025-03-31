"""Tests for the SummarizerAgent."""

from unittest import mock

from portia.execution_agents.output import Output
from portia.execution_agents.utils.final_output_summarizer import FinalOutputSummarizer
from portia.introspection_agents.introspection_agent import PreStepIntrospectionOutcome
from portia.plan import Step
from tests.utils import get_test_config, get_test_plan_run


def test_summarizer_agent_execute_sync() -> None:
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
        "$london_weather": Output(value="Sunny and warm"),
        "$activities": Output(value="Visit Hyde Park and have a picnic"),
    }

    # Mock LLM response
    expected_summary = "Weather is sunny and warm in London, visit to Hyde Park for a picnic"
    mock_llm = mock.MagicMock()
    mock_llm.invoke.return_value.content = expected_summary

    with mock.patch(
        "portia.execution_agents.utils.final_output_summarizer.LLMWrapper.for_usage",
    ) as mock_wrapper:
        mock_wrapper.return_value.to_langchain.return_value = mock_llm

        summarizer = FinalOutputSummarizer(config=get_test_config())
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
        expected_prompt = FinalOutputSummarizer.SUMMARIZE_TASK + expected_context
        mock_llm.invoke.assert_called_once_with(expected_prompt)


def test_summarizer_agent_empty_plan_run() -> None:
    """Test summarizer agent with empty plan run."""
    (plan, plan_run) = get_test_plan_run()
    plan.plan_context.query = "Empty query"
    plan.steps = []
    plan_run.outputs.step_outputs = {}

    mock_llm = mock.MagicMock()
    mock_llm.invoke.return_value.content = "Empty summary"

    with mock.patch(
        "portia.execution_agents.utils.final_output_summarizer.LLMWrapper.for_usage",
    ) as mock_wrapper:
        mock_wrapper.return_value.to_langchain.return_value = mock_llm

        summarizer = FinalOutputSummarizer(config=get_test_config())
        output = summarizer.create_summary(plan=plan, plan_run=plan_run)

        # Verify empty context case
        assert output == "Empty summary"
        expected_prompt = FinalOutputSummarizer.SUMMARIZE_TASK + ("Query: Empty query\n----------")
        mock_llm.invoke.assert_called_once_with(expected_prompt)


def test_summarizer_agent_handles_none_response() -> None:
    """Test that the agent handles None response from LLM."""
    (plan, plan_run) = get_test_plan_run()
    plan.plan_context.query = "Test query"

    mock_llm = mock.MagicMock()
    mock_llm.invoke.return_value.content = None

    with mock.patch(
        "portia.execution_agents.utils.final_output_summarizer.LLMWrapper.for_usage",
    ) as mock_wrapper:
        mock_wrapper.return_value.to_langchain.return_value = mock_llm

        summarizer = FinalOutputSummarizer(config=get_test_config())
        output = summarizer.create_summary(plan=plan, plan_run=plan_run)

        # Verify None handling
        assert output is None


def test_build_tasks_and_outputs_context() -> None:
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
        "$london_weather": Output(value="Sunny and warm"),
        "$activities": Output(value="Visit Hyde Park and have a picnic"),
    }

    summarizer = FinalOutputSummarizer(config=get_test_config())
    context = summarizer._build_tasks_and_outputs_context(  # noqa: SLF001
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
    context = summarizer._build_tasks_and_outputs_context(  # noqa: SLF001
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
        "$london_weather": Output(value="Sunny and warm"),
    }

    summarizer = FinalOutputSummarizer(config=get_test_config())
    context = summarizer._build_tasks_and_outputs_context(  # noqa: SLF001
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
            task="Failed task",
            output="$failed_output",
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
        "$regular_output": Output(value="Regular result", summary="Not used"),
        "$failed_output": Output(
            value=PreStepIntrospectionOutcome.FAIL,
            summary="This task failed due to an error",
        ),
        "$skipped_output": Output(
            value=PreStepIntrospectionOutcome.SKIP,
            summary="This task was skipped as it was unnecessary",
        ),
        "$complete_output": Output(
            value=PreStepIntrospectionOutcome.COMPLETE,
            summary="The plan execution was completed early",
        ),
    }

    summarizer = FinalOutputSummarizer(config=get_test_config())
    context = summarizer._build_tasks_and_outputs_context(  # noqa: SLF001
        plan=plan,
        plan_run=plan_run,
    )

    assert context == (
        "Query: Test query with conditional outcomes\n"
        "----------\n"
        "Task: Regular task\n"
        "Output: Regular result\n"
        "----------\n"
        "Task: Failed task\n"
        "Output: This task failed due to an error\n"
        "----------\n"
        "Task: Skipped task\n"
        "Output: This task was skipped as it was unnecessary\n"
        "----------\n"
        "Task: Complete task\n"
        "Output: The plan execution was completed early\n"
        "----------"
    )
