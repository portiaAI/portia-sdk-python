"""Tests for the Planner module."""

from unittest.mock import MagicMock

import pytest

from portia.config import Config
from portia.llm_wrapper import LLMWrapper
from portia.plan import Plan
from portia.planner import Planner, PlanOrError
from portia.tool_registry import ToolSet


@pytest.fixture
def mock_config() -> Config:
    """Mock Config object for testing."""
    return MagicMock()


@pytest.fixture
def mock_tool_set() -> ToolSet:
    """Mock ToolSet object for testing."""
    return MagicMock()


@pytest.fixture
def planner(mock_config: Config) -> Planner:
    """Create an instance of the Planner with mocked config."""
    return Planner(config=mock_config)


def test_generate_plan_or_error_success(planner: Planner, mock_tool_set: ToolSet) -> None:
    """Test successful plan generation with valid inputs."""
    mock_tool_set.get_tools.return_value = []  # type: ignore  # noqa: PGH003
    query = "Send hello@portialabs.ai an email with a summary of the latest news on AI"

    # Mock the LLMWrapper response to simulate a successful plan generation
    mock_response = PlanOrError(plan=Plan(query=query, steps=[]), error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    result = planner.generate_plan_or_error(query=query, tool_list=mock_tool_set)

    assert result.plan.query == query
    assert result.error is None
