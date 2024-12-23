"""Tests for runner classes."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from portia.config import AgentType, Config, default_config
from portia.errors import InvalidWorkflowStateError
from portia.llm_wrapper import LLMWrapper
from portia.plan import Plan
from portia.planner import PlanOrError
from portia.runner import Runner
from portia.tool_registry import InMemoryToolRegistry
from portia.workflow import WorkflowState
from tests.utils import AdditionTool, ClarificationTool


@pytest.fixture
def runner() -> Runner:
    """Fixture to create a Runner instance for testing."""
    config = default_config()
    tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool(), ClarificationTool()])
    return Runner(config=config, tool_registry=tool_registry)


def test_runner_run_query(runner: Runner) -> None:
    """Test running a query using the Runner."""
    query = "example query"

    mock_response = PlanOrError(plan=Plan(query=query, steps=[]), error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    workflow = runner.run_query(query)

    assert workflow.state == WorkflowState.COMPLETE


def test_runner_plan_query(runner: Runner) -> None:
    """Test planning a query using the Runner."""
    query = "example query"

    mock_response = PlanOrError(plan=Plan(query=query, steps=[]), error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.plan_query(query)

    assert plan.query == query


def test_runner_run_plan(runner: Runner) -> None:
    """Test running a plan using the Runner."""
    query = "example query"

    mock_response = PlanOrError(plan=Plan(query=query, steps=[]), error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.plan_query(query)
    workflow = runner.run_plan(plan)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.plan_id == plan.id


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
    config_data = """{
"portia_api_key": "file-key",
"openai_api_key": "file-openai-key",
"llm_model_temperature": 10,
"storage_class": "MEMORY",
"llm_provider": "OPENAI",
"llm_model_name": "gpt-4o-mini",
"llm_model_seed": 443,
"default_agent_type": "VERIFIER"
}"""

    with tempfile.NamedTemporaryFile("w", delete=True, suffix=".json") as temp_file:
        temp_file.write(config_data)
        temp_file.flush()

        config_file = Path(temp_file.name)

        config = Config.from_file(config_file)

        assert config.must_get_raw_api_key("portia_api_key") == "file-key"
        assert config.must_get_raw_api_key("openai_api_key") == "file-openai-key"
        assert config.default_agent_type == AgentType.VERIFIER
        assert config.llm_model_temperature == 10
