"""Tests for the Planner module."""

from unittest.mock import MagicMock
from uuid import UUID

import pytest

from portia.config import Config
from portia.llm_wrapper import LLMWrapper
from portia.plan import Plan
from portia.planner import Planner, PlanOrError


@pytest.fixture
def mock_config() -> Config:
    """Mock Config object for testing."""
    return MagicMock()


@pytest.fixture
def planner(mock_config: Config) -> Planner:
    """Create an instance of the Planner with mocked config."""
    return Planner(llm_wrapper=LLMWrapper(config=mock_config))


def test_plan_uuid_assign() -> None:
    """Test plan assign correct UUIDs."""
    plan = Plan(
        id="123",  # type: ignore  # noqa: PGH003
        query="",
        steps=[],
    )
    assert isinstance(plan.id, UUID)

    clarification = Plan(
        id=123,  # type: ignore  # noqa: PGH003
        query="",
        steps=[],
    )
    assert isinstance(clarification.id, UUID)


def test_generate_plan_or_error_success(planner: Planner) -> None:
    """Test successful plan generation with valid inputs."""
    query = "Send hello@portialabs.ai an email with a summary of the latest news on AI"

    # Mock the LLMWrapper response to simulate a successful plan generation
    mock_response = PlanOrError(plan=Plan(query=query, steps=[]), error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    result = planner.generate_plan_or_error(query=query, tool_list=[])

    assert result.plan.query == query
    assert result.error is None


def test_generate_plan_or_error_failure(planner: Planner) -> None:
    """Test handling of error when generating a plan fails."""
    query = "Send hello@portialabs.ai an email with a summary of the latest news on AI"

    # Mock the LLMWrapper response to simulate an error in plan generation
    mock_response = PlanOrError(plan=Plan(query=query, steps=[]), error="Unable to generate a plan")
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    result = planner.generate_plan_or_error(query=query, tool_list=[])

    assert result.error == "Unable to generate a plan"
    assert result.plan.query == query
