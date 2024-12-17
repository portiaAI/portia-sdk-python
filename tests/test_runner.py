"""Tests for runner classes."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from portia.config import Config
from portia.llm_wrapper import LLMWrapper
from portia.plan import Plan
from portia.planner import PlanOrError
from portia.runner import Runner
from portia.tool import Tool
from portia.tool_registry import LocalToolRegistry
from portia.workflow import InvalidWorkflowStateError, WorkflowState


class AdditionTool(Tool):
    """Add numbers."""

    id: str = "add_tool"
    name: str = "Add Tool"
    description: str = "Takes two numbers and adds them together"

    def run(self, a: int, b: int) -> int:
        """Add the numbers."""
        return a + b


@pytest.fixture
def runner() -> Runner:
    """Fixture to create a Runner instance for testing."""
    config = Config()
    tool_registry = LocalToolRegistry.from_local_tools([AdditionTool()])
    return Runner(config=config, tool_registry=tool_registry)


@patch.object(LLMWrapper, "_instance", None)
def test_runner_run_query(runner: Runner) -> None:
    """Test running a query using the Runner."""
    query = "example query"

    mock_response = PlanOrError(plan=Plan(query=query, steps=[]), error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    workflow = runner.run_query(query)

    assert workflow.state == WorkflowState.COMPLETE


@patch.object(LLMWrapper, "_instance", None)
def test_runner_plan_query(runner: Runner) -> None:
    """Test planning a query using the Runner."""
    query = "example query"

    mock_response = PlanOrError(plan=Plan(query=query, steps=[]), error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.plan_query(query)

    assert plan.query == query


@patch.object(LLMWrapper, "_instance", None)
def test_runner_run_plan(runner: Runner) -> None:
    """Test running a plan using the Runner."""
    query = "example query"

    mock_response = PlanOrError(plan=Plan(query=query, steps=[]), error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.plan_query(query)
    workflow = runner.run_plan(plan)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.plan_id == plan.id


@patch.object(LLMWrapper, "_instance", None)
def test_runner_resume_workflow(runner: Runner) -> None:
    """Test resuming a workflow after interruption."""
    query = "example query"

    mock_response = PlanOrError(plan=Plan(query=query, steps=[]), error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.plan_query(query)
    workflow = runner.run_plan(plan)

    # Simulate workflow being in progress
    workflow.state = WorkflowState.IN_PROGRESS
    workflow.current_step_index = 1
    workflow = runner.resume_workflow(workflow)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.current_step_index == 1


@patch.object(LLMWrapper, "_instance", None)
def test_runner_resume_workflow_invalid_state(runner: Runner) -> None:
    """Test resuming a workflow with an invalid state."""
    query = "example query"

    mock_response = PlanOrError(plan=Plan(query=query, steps=[]), error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.plan_query(query)
    workflow = runner.run_plan(plan)

    # Set invalid state
    workflow.state = WorkflowState.COMPLETE

    with pytest.raises(InvalidWorkflowStateError):
        runner.resume_workflow(workflow)


def test_runner_config_from_file() -> None:
    """Test loading configuration from a file."""
    config_data = '{"portia_api_key": "file-key", "openai_api_key": "file-openai-key", "llm_model_temperature": 10}'  # noqa: E501
    config_file = Path("config.json")
    try:
        config_file.write_text(config_data)

        config = Config.from_file(config_file)

        assert config.must_get_raw_api_key("portia_api_key") == "file-key"
        assert config.must_get_raw_api_key("openai_api_key") == "file-openai-key"
        assert config.llm_model_temperature == 10
    finally:
        if config_file.exists():
            config_file.unlink()  # Remove the file after the test
