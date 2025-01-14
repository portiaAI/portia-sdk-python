"""Tests for runner classes."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from portia.config import AgentType, StorageClass
from portia.errors import InvalidWorkflowStateError, PlanError
from portia.llm_wrapper import LLMWrapper
from portia.plan import Step
from portia.planner import StepsOrError
from portia.runner import Runner
from portia.tool_registry import InMemoryToolRegistry
from portia.workflow import WorkflowState
from tests.utils import AdditionTool, ClarificationTool, get_test_config


@pytest.fixture
def runner() -> Runner:
    """Fixture to create a Runner instance for testing."""
    config = get_test_config()
    tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool(), ClarificationTool()])
    return Runner(config=config, tool_registry=tool_registry)


def test_runner_run_query(runner: Runner) -> None:
    """Test running a query using the Runner."""
    query = "example query"

    mock_response = StepsOrError(
        steps=[],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    workflow = runner.run_query(query)

    assert workflow.state == WorkflowState.COMPLETE


def test_runner_run_query_disk_storage() -> None:
    """Test running a query using the Runner."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        query = "example query"
        config = get_test_config(
            storage_class=StorageClass.DISK,
            storage_dir=tmp_dir,
        )
        tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool(), ClarificationTool()])
        runner = Runner(config=config, tool_registry=tool_registry)

        mock_response = StepsOrError(steps=[], error=None)
        LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

        workflow = runner.run_query(query)

        assert workflow.state == WorkflowState.COMPLETE
        # Use Path to check for the files
        plan_files = list(Path(tmp_dir).glob("plan-*.json"))
        workflow_files = list(Path(tmp_dir).glob("workflow-*.json"))

        assert len(plan_files) == 1
        assert len(workflow_files) == 1


def test_runner_plan_query(runner: Runner) -> None:
    """Test planning a query using the Runner."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.plan_query(query)

    assert plan.plan_context.query == query


def test_runner_plan_query_error(runner: Runner) -> None:
    """Test planning a query that returns an error."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error="could not plan")
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    with pytest.raises(PlanError):
        runner.plan_query(query)


def test_runner_plan_query_with_tools(runner: Runner) -> None:
    """Test planning a query using the Runner."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.plan_query(query, tools=["Add Tool"])

    assert plan.plan_context.query == query
    assert plan.plan_context.tool_list == ["add_tool"]


def test_runner_create_and_execute_workflow(runner: Runner) -> None:
    """Test running a plan using the Runner."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.plan_query(query)
    workflow = runner.create_and_execute_workflow(plan)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.plan_id == plan.id


def test_runner_toolless_agent() -> None:
    """Test running a plan using the Runner."""
    query = "example query"

    mock_response = StepsOrError(
        steps=[
            Step(
                task="Find and summarize the latest news on artificial intelligence",
                tool_name="Add Tool",
                output="$ai_search_results",
            ),
        ],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    config = get_test_config(
        default_agent_type=AgentType.TOOL_LESS,
    )
    tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool(), ClarificationTool()])
    runner = Runner(config=config, tool_registry=tool_registry)

    plan = runner.plan_query(query)
    runner.create_and_execute_workflow(plan)


def test_runner_execute_workflow(runner: Runner) -> None:
    """Test resuming a workflow after interruption."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.plan_query(query)
    workflow = runner.create_and_execute_workflow(plan)

    # Simulate workflow being in progress
    workflow.state = WorkflowState.IN_PROGRESS
    workflow.current_step_index = 1
    workflow = runner.execute_workflow(workflow)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.current_step_index == 1


def test_runner_execute_workflow_invalid_state(runner: Runner) -> None:
    """Test resuming a workflow with an invalid state."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.plan_query(query)
    workflow = runner.create_and_execute_workflow(plan)

    # Set invalid state
    workflow.state = WorkflowState.COMPLETE

    with pytest.raises(InvalidWorkflowStateError):
        runner.execute_workflow(workflow)
