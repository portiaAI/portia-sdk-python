"""Tests for runner classes."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from portia.agents.base_agent import Output
from portia.config import AgentType, StorageClass
from portia.errors import InvalidWorkflowStateError, PlanError, WorkflowNotFoundError
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

    workflow = runner.execute_query(query)

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

        workflow = runner.execute_query(query)

        assert workflow.state == WorkflowState.COMPLETE
        # Use Path to check for the files
        plan_files = list(Path(tmp_dir).glob("plan-*.json"))
        workflow_files = list(Path(tmp_dir).glob("workflow-*.json"))

        assert len(plan_files) == 1
        assert len(workflow_files) == 1


def test_runner_generate_plan(runner: Runner) -> None:
    """Test planning a query using the Runner."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.generate_plan(query)

    assert plan.plan_context.query == query


def test_runner_generate_plan_error(runner: Runner) -> None:
    """Test planning a query that returns an error."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error="could not plan")
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    with pytest.raises(PlanError):
        runner.generate_plan(query)


def test_runner_generate_plan_with_tools(runner: Runner) -> None:
    """Test planning a query using the Runner."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.generate_plan(query, tools=["Add Tool"])

    assert plan.plan_context.query == query
    assert plan.plan_context.tool_ids == ["add_tool"]


def test_runner_create_and_execute_workflow(runner: Runner) -> None:
    """Test running a plan using the Runner."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.generate_plan(query)
    workflow = runner.create_workflow(plan)
    workflow = runner.execute_workflow(workflow)

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

    plan = runner.generate_plan(query)
    workflow = runner.create_workflow(plan)
    workflow = runner.execute_workflow(workflow)


def test_runner_execute_workflow(runner: Runner) -> None:
    """Test resuming a workflow after interruption."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.generate_plan(query)
    workflow = runner.create_workflow(plan)
    workflow = runner.execute_workflow(workflow)

    # Simulate workflow being in progress
    workflow.state = WorkflowState.IN_PROGRESS
    workflow.current_step_index = 1
    workflow = runner.execute_workflow(workflow)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.current_step_index == 1


def test_runner_execute_workflow_edge_cases(runner: Runner) -> None:
    """Test edge cases for execute."""
    with pytest.raises(ValueError):  # noqa: PT011
        runner.execute_workflow()

    query = "example query"
    mock_response = StepsOrError(
        steps=[],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.generate_plan(query)
    workflow = runner.create_workflow(plan)

    # Simulate workflow being in progress
    workflow.state = WorkflowState.IN_PROGRESS
    workflow.current_step_index = 1
    workflow = runner.execute_workflow(workflow_id=workflow.id)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.current_step_index == 1

    with pytest.raises(WorkflowNotFoundError):
        runner.execute_workflow(workflow_id=uuid4())


def test_runner_execute_workflow_invalid_state(runner: Runner) -> None:
    """Test resuming a workflow with an invalid state."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.generate_plan(query)
    workflow = runner.create_workflow(plan)
    workflow = runner.execute_workflow(workflow)

    # Set invalid state
    workflow.state = WorkflowState.COMPLETE

    with pytest.raises(InvalidWorkflowStateError):
        runner.execute_workflow(workflow)


def test_runner_sets_final_output_correctly(runner: Runner) -> None:
    """Test that final output is set correctly with summary from last step."""
    query = "What activities can I do in Cairo based on weather?"

    # Mock planner to return 2 steps
    mock_plan_response = StepsOrError(
        steps=[
            Step(
                task="Get current weather in Cairo",
                tool_name="Add Tool",
                output="$weather",
            ),
            Step(
                task="Suggest activities based on weather",
                tool_name="Add Tool",
                output="$activities",
            ),
        ],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_plan_response)

    # Create plan and workflow
    plan = runner.generate_plan(query)
    workflow = runner.create_workflow(plan)

    # Mock agent responses for each step
    weather_response = "Sunny and 75°F in Cairo"
    activities_response = "Perfect weather for visiting the pyramids and walking along the Nile"
    final_summary = "Cairo has 75°F weather, you can visit the pyramids and walk along the Nile"

    mock_agent = MagicMock()
    mock_agent.execute_sync.side_effect = [
        Output(value=weather_response),
        Output(value=activities_response),
        Output(value=final_summary),
    ]
    runner._get_agent_for_step = MagicMock(return_value=mock_agent) # noqa: SLF001

    # Execute workflow
    workflow = runner.execute_workflow(workflow)

    # Verify outputs
    assert workflow.outputs.step_outputs["$weather"].value == weather_response
    assert workflow.outputs.step_outputs["$activities"].value == activities_response

    # Verify final output
    assert workflow.outputs.final_output.value == final_summary # pyright: ignore[reportOptionalMemberAccess]
    assert workflow.outputs.final_output.summary == final_summary  # pyright: ignore[reportOptionalMemberAccess]
    assert workflow.state == WorkflowState.COMPLETE
