"""Tests for the SummarizerAgent."""

from unittest import mock

from portia.agents.base_agent import Output
from portia.agents.utils.summarizer_agent import SummarizerAgent
from portia.plan import Step
from tests.utils import get_test_config, get_test_workflow


def test_summarizer_agent_execute_sync() -> None:
    """Test that the summarizer agent correctly executes and returns a summary."""
    # Set up test data
    (plan, workflow) = get_test_workflow()
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

    with mock.patch("portia.agents.utils.summarizer_agent.LLMWrapper") as mock_wrapper:
        mock_wrapper.return_value.to_langchain.return_value = mock_llm

        agent = SummarizerAgent(workflow=workflow, plan=plan, config=get_test_config())
        output = agent.execute_sync()

        # Verify the output
        assert output.value == expected_summary
        assert output.summary == expected_summary

        # Verify LLM was called with correct prompt
        expected_context = (
            "Task: Get weather in London\n"
            "Output: Sunny and warm\n"
            "----------\n"
            "Task: Suggest activities based on weather\n"
            "Output: Visit Hyde Park and have a picnic\n"
            "----------"
        )
        expected_prompt = SummarizerAgent.SUMMARIZE_TASK + expected_context
        mock_llm.invoke.assert_called_once_with(expected_prompt)


def test_summarizer_agent_empty_workflow() -> None:
    """Test summarizer agent with empty workflow."""
    (plan, workflow) = get_test_workflow()
    plan.steps = []
    workflow.outputs.step_outputs = {}

    mock_llm = mock.MagicMock()
    mock_llm.invoke.return_value.content = "Empty summary"

    with mock.patch("portia.agents.utils.summarizer_agent.LLMWrapper") as mock_wrapper:
        mock_wrapper.return_value.to_langchain.return_value = mock_llm

        agent = SummarizerAgent(workflow=workflow, plan=plan, config=get_test_config())
        output = agent.execute_sync()

        # Verify empty context case
        assert output.value == "Empty summary"
        mock_llm.invoke.assert_called_once_with(SummarizerAgent.SUMMARIZE_TASK)


def test_summarizer_agent_handles_none_response() -> None:
    """Test that the agent handles None response from LLM."""
    (plan, workflow) = get_test_workflow()

    mock_llm = mock.MagicMock()
    mock_llm.invoke.return_value.content = None

    with mock.patch("portia.agents.utils.summarizer_agent.LLMWrapper") as mock_wrapper:
        mock_wrapper.return_value.to_langchain.return_value = mock_llm

        agent = SummarizerAgent(workflow=workflow, plan=plan, config=get_test_config())
        output = agent.execute_sync()

        # Verify None handling
        assert output.value is None
        assert output.summary is None
