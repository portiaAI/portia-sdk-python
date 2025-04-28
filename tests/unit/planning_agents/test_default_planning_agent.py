"""Tests for the PlanningAgent module."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from portia.end_user import EndUser
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import Plan, PlanContext, PlanInput, Step, Variable
from portia.planning_agents.base_planning_agent import BasePlanningAgent, StepsOrError
from portia.planning_agents.context import (
    render_prompt_insert_defaults,
)
from portia.planning_agents.default_planning_agent import DefaultPlanningAgent
from tests.utils import AdditionTool, get_mock_generative_model, get_test_config

if TYPE_CHECKING:
    from portia.config import Config
    from portia.tool import Tool


@pytest.fixture
def mock_config() -> Config:
    """Mock Config object for testing."""
    return MagicMock()


def test_generate_steps_or_error_success(mock_config: Config) -> None:
    """Test successful plan generation with valid inputs."""
    query = "Send hello@portialabs.ai an email with a summary of the latest news on AI"

    # Mock the Model response to simulate a successful plan generation
    mock_model = get_mock_generative_model(
        response=StepsOrError(
            steps=[],
            error=None,
        ),
    )
    mock_config.get_planning_model.return_value = mock_model  # type: ignore[reportFunctionMemberAccess]
    planning_agent = DefaultPlanningAgent(mock_config)

    result = planning_agent.generate_steps_or_error(
        query=query,
        tool_list=[],
        end_user=EndUser(external_id="123"),
    )

    assert result.steps == []
    assert result.error is None


def test_base_classes() -> None:
    """Test PlanStorage raises."""

    class MyPlanningAgent(BasePlanningAgent):
        """Override to test base."""

        def generate_steps_or_error(
            self,
            query: str,
            tool_list: list[Tool],
            end_user: EndUser,
            examples: list[Plan] | None = None,
        ) -> StepsOrError:
            return super().generate_steps_or_error(query, tool_list, end_user, examples)  # type: ignore  # noqa: PGH003

    wrapper = MyPlanningAgent(get_test_config())

    with pytest.raises(NotImplementedError):
        wrapper.generate_steps_or_error("", [], EndUser(external_id="123"), [])


def test_generate_steps_or_error_failure(mock_config: Config) -> None:
    """Test handling of error when generating a plan fails."""
    query = "Send hello@portialabs.ai an email with a summary of the latest news on AI"

    # Mock the Model response to simulate an error in plan generation
    mock_model = get_mock_generative_model(
        response=StepsOrError(
            steps=[],
            error="Unable to generate a plan",
        ),
    )
    mock_config.get_planning_model.return_value = mock_model  # type: ignore[reportFunctionMemberAccess]
    planning_agent = DefaultPlanningAgent(mock_config)
    result = planning_agent.generate_steps_or_error(
        query=query,
        tool_list=[],
        end_user=EndUser(external_id="123"),
    )

    assert result.error == "Unable to generate a plan"


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
                    tool_id="plan_tool1a",
                    inputs=[Variable(name="$plan_input1", description="plan description 1")],
                    output="$plan_output1",
                ),
            ],
            inputs=[
                PlanInput(name="$plan_input", description="Plan input description"),
            ],
        ),
    ]
    rendered_prompt = render_prompt_insert_defaults(
        query="test query",
        tool_list=[AdditionTool()],
        examples=plans,
        end_user=EndUser(external_id="123"),
    )
    overall_pattern = re.compile(
        r"<Example>(.*?)</Example>.*?<Tools>(.*?)</Tools>.*?<Request>(.*?)</Request>.*?",
        re.DOTALL,
    )
    example_match, tools_content, request_content = overall_pattern.findall(
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

    # Check that plan inputs are included in the example
    assert "$plan_input" in example_match
    assert "Plan input description" in example_match

    assert "Use this tool to add two numbers together" in tools_content
    assert "add_tool" in tools_content

    assert "test query" in request_content


def test_generate_steps_or_error_invalid_tool_id(mock_config: Config) -> None:
    """Test handling of invalid tool ID in generated steps."""
    query = "Calculate something"

    mock_response = StepsOrError(
        steps=[
            Step(
                task="Calculate sum",
                tool_id="no_tool_1",
                inputs=[],
                output="$result",
            ),
            Step(
                task="Calculate sum2",
                tool_id="no_tool_2",
                inputs=[],
                output="$result2",
            ),
        ],
        error=None,
    )
    mock_model = get_mock_generative_model(
        response=mock_response,
    )
    mock_config.get_planning_model.return_value = mock_model  # type: ignore[reportFunctionMemberAccess]
    planning_agent = DefaultPlanningAgent(mock_config)
    result = planning_agent.generate_steps_or_error(
        query=query,
        tool_list=[AdditionTool()],
        end_user=EndUser(external_id="123"),
    )

    assert result.error == "Missing tools no_tool_1, no_tool_2 from the provided tool_list"
    assert result.steps == mock_response.steps


def test_generate_steps_assigns_llm_tool_id(mock_config: Config) -> None:
    """Test that steps without tool_id get assigned to LLMTool."""
    query = "Generate a creative story"

    # Mock response with steps that have no tool_id
    mock_response = StepsOrError(
        steps=[
            Step(
                task="Write a story opening",
                tool_id=None,
                inputs=[],
                output="$story_opening",
            ),
            Step(
                task="Write story conclusion",
                tool_id=None,
                inputs=[],
                output="$story_conclusion",
            ),
        ],
        error=None,
    )
    mock_model = get_mock_generative_model(
        response=mock_response,
    )
    mock_config.get_planning_model.return_value = mock_model  # type: ignore[reportFunctionMemberAccess]
    planning_agent = DefaultPlanningAgent(mock_config)
    result = planning_agent.generate_steps_or_error(
        query=query,
        tool_list=[AdditionTool()],
        end_user=EndUser(external_id="123"),
    )

    assert all(step.tool_id == LLMTool.LLM_TOOL_ID for step in result.steps)
    assert len(result.steps) == 2
    assert result.error is None


def test_generate_steps_with_plan_inputs(mock_config: Config) -> None:
    """Test plan generation with plan inputs."""
    plan_inputs = [
        PlanInput(
            name="$username",
            description="Username for the service",
        ),
        PlanInput(
            name="$user_id",
            description="ID of the user",
        ),
    ]

    mock_model = get_mock_generative_model(
        response=StepsOrError(
            steps=[
                Step(
                    task="Process user addition",
                    tool_id="add_tool",
                    inputs=[Variable(name="$user_id", description="ID of the user")],
                    output="$output",
                ),
            ],
            error=None,
        ),
    )
    mock_config.get_planning_model.return_value = mock_model  # type: ignore[reportFunctionMemberAccess]
    planning_agent = DefaultPlanningAgent(mock_config)

    result = planning_agent.generate_steps_or_error(
        query="Process user data",
        tool_list=[AdditionTool()],
        end_user=EndUser(external_id="123"),
        plan_inputs=plan_inputs,
    )

    assert mock_model.get_structured_response.called
    messages = mock_model.get_structured_response.call_args[1]["messages"]
    prompt_text = messages[1].content

    assert "$user_id" in prompt_text
    assert "ID of the user" in prompt_text
    assert result.error is None
