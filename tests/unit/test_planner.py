"""Tests for the Planner module."""

import re
from unittest.mock import MagicMock

import pytest

from portia.config import Config
from portia.example_tools.addition import AdditionTool
from portia.llm_wrapper import LLMWrapper
from portia.plan import Plan, PlanContext, Step, Variable
from portia.planner import (
    Planner,
    StepsOrError,
    _default_query_system_context,
    _render_prompt_insert_defaults,
)


@pytest.fixture
def mock_config() -> Config:
    """Mock Config object for testing."""
    return MagicMock()


@pytest.fixture
def planner(mock_config: Config) -> Planner:
    """Create an instance of the Planner with mocked config."""
    return Planner(llm_wrapper=LLMWrapper(config=mock_config))


def test_generate_plan_or_error_success(planner: Planner) -> None:
    """Test successful plan generation with valid inputs."""
    query = "Send hello@portialabs.ai an email with a summary of the latest news on AI"

    # Mock the LLMWrapper response to simulate a successful plan generation
    mock_response = StepsOrError(
        steps=[],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    result = planner.generate_plan_or_error(query=query, tool_list=[])

    assert result.plan.plan_context.query == query
    assert result.error is None


def test_generate_plan_or_error_failure(planner: Planner) -> None:
    """Test handling of error when generating a plan fails."""
    query = "Send hello@portialabs.ai an email with a summary of the latest news on AI"

    # Mock the LLMWrapper response to simulate an error in plan generation
    mock_response = StepsOrError(
        steps=[],
        error="Unable to generate a plan",
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    result = planner.generate_plan_or_error(query=query, tool_list=[])

    assert result.error == "Unable to generate a plan"
    assert result.plan.plan_context.query == query


def test_planner_default_context_with_extensions() -> None:
    """Test default context."""
    context = _default_query_system_context(system_context_extension=["456"])
    assert "456" in context


def test_render_prompt() -> None:
    """Test render prompt."""
    plans = [
        Plan(
            plan_context=PlanContext(
                query="plan query 1",
                tool_ids=["plan_tool1a", "plan_tool1b"],
            ),
            steps=[
                Step(
                    task="plan task 1",
                    tool_name="plan_tool1a",
                    inputs=[Variable(name="$plan_input1", description="plan description 1")],
                    output="$plan_output1",
                ),
            ],
        ),
    ]
    rendered_prompt = _render_prompt_insert_defaults(
        query="test query",
        tool_list=[AdditionTool()],
        examples=plans,
        system_context_extension=["extension"],
    )
    overall_pattern = re.compile(
        r"<Example>(.*?)</Example>.*?<Tools>(.*?)</Tools>.*?<Request>(.*?)</Request>.*?"
        r"<SystemContext>(.*?)</SystemContext>",
        re.DOTALL,
    )
    example_match, tools_content, request_content, system_context_content = overall_pattern.findall(
        rendered_prompt,
    )[0]

    tool_pattern = re.compile(r"<Tools>(.*?)</Tools>", re.DOTALL)
    tool_match = tool_pattern.findall(example_match)[0]

    assert "plan_tool1a" in tool_match
    assert "plan_tool1b" in tool_match

    query_pattern = re.compile(r"<Query>(.*?)</Query>", re.DOTALL)
    query_match = query_pattern.findall(example_match)[0]

    assert "plan query 1" in query_match

    response_pattern = re.compile(r"<Response>(.*?)</Response>", re.DOTALL)
    response_match = response_pattern.findall(example_match)[0]

    assert "plan task 1" in response_match
    assert "plan_tool1a" in response_match
    assert "$plan_input1" in response_match
    assert "$plan_output1" in response_match

    assert "Takes two float and adds them together" in tools_content
    assert "Add Tool" in tools_content

    assert "test query" in request_content
    assert "Add Tool" in request_content
    assert "extension" in system_context_content
