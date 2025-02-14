"""Tests for the Planner module."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from langchain_openai import ChatOpenAI

from portia.execution_context import ExecutionContext, get_execution_context
from portia.plan import Plan, PlanContext, Step, Variable
from portia.planners.context import default_query_system_context, render_prompt_insert_defaults
from portia.planners.one_shot_planner import OneShotPlanner
from portia.planners.planner import Planner, StepsOrError
from tests.utils import AdditionTool, MockInvoker, get_test_config

if TYPE_CHECKING:
    from portia.tool import Tool


def test_generate_steps_or_error_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful plan generation with valid inputs."""
    query = "Send hello@portialabs.ai an email with a summary of the latest news on AI"

    # Mock the LLMWrapper response to simulate a successful plan generation
    mock_response = StepsOrError(
        steps=[],
        error=None,
    )

    mock_invoker = MockInvoker(response=mock_response)
    monkeypatch.setattr(ChatOpenAI, "invoke", mock_invoker.invoke)
    monkeypatch.setattr(ChatOpenAI, "with_structured_output", mock_invoker.with_structured_output)

    planner = OneShotPlanner(config=get_test_config())

    result = planner.generate_steps_or_error(
        ctx=get_execution_context(),
        query=query,
        tool_list=[],
    )

    assert result.steps == []
    assert result.error is None


def test_base_classes() -> None:
    """Test PlanStorage raises."""

    class MyPlanner(Planner):
        """Override to test base."""

        def generate_steps_or_error(
            self,
            ctx: ExecutionContext,
            query: str,
            tool_list: list[Tool],
            examples: list[Plan] | None = None,
        ) -> StepsOrError:
            return super().generate_steps_or_error(ctx, query, tool_list, examples)  # type: ignore  # noqa: PGH003

    wrapper = MyPlanner(get_test_config())

    with pytest.raises(NotImplementedError):
        wrapper.generate_steps_or_error(get_execution_context(), "", [], [])


def test_generate_steps_or_error_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of error when generating a plan fails."""
    query = "Send hello@portialabs.ai an email with a summary of the latest news on AI"

    # Mock the LLMWrapper response to simulate an error in plan generation
    mock_response = StepsOrError(
        steps=[],
        error="Unable to generate a plan",
    )
    mock_invoker = MockInvoker(response=mock_response)
    monkeypatch.setattr(ChatOpenAI, "invoke", mock_invoker.invoke)
    monkeypatch.setattr(ChatOpenAI, "with_structured_output", mock_invoker.with_structured_output)

    planner = OneShotPlanner(config=get_test_config())

    result = planner.generate_steps_or_error(ctx=get_execution_context(), query=query, tool_list=[])

    assert result.error == "Unable to generate a plan"


def test_planner_default_context_with_extensions() -> None:
    """Test default context."""
    context = default_query_system_context(system_context_extension=["456"])
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
                    tool_id="plan_tool1a",
                    inputs=[Variable(name="$plan_input1", description="plan description 1")],
                    output="$plan_output1",
                ),
            ],
        ),
    ]
    rendered_prompt = render_prompt_insert_defaults(
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

    assert "Takes two numbers and adds them together" in tools_content
    assert "add_tool" in tools_content

    assert "test query" in request_content
    assert "add_tool" in request_content
    assert "extension" in system_context_content
