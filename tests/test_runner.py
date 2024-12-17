"""Tests for runner classes."""

from pathlib import Path

import pytest

from portia.runner import Runner, RunnerConfig
from portia.storage import InMemoryStorage
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
    config = RunnerConfig()
    storage = InMemoryStorage()
    tool_registry = LocalToolRegistry.from_local_tools([AdditionTool()])
    return Runner(config=config, storage=storage, tool_registry=tool_registry)


def test_runner_run_query(runner: Runner) -> None:
    """Test running a query using the Runner."""
    query = "example query"
    workflow = runner.run_query(query)

    assert workflow.state == WorkflowState.COMPLETE


def test_runner_plan_query(runner: Runner) -> None:
    """Test planning a query using the Runner."""
    query = "example query"
    plan = runner.plan_query(query)

    assert plan.query == query
    assert len(plan.steps) > 0


def test_runner_run_plan(runner: Runner) -> None:
    """Test running a plan using the Runner."""
    query = "example query"
    plan = runner.plan_query(query)
    workflow = runner.run_plan(plan)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.plan_id == plan.id


def test_runner_resume_workflow(runner: Runner) -> None:
    """Test resuming a workflow after interruption."""
    query = "example query"
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
    plan = runner.plan_query(query)
    workflow = runner.run_plan(plan)

    # Set invalid state
    workflow.state = WorkflowState.COMPLETE

    with pytest.raises(InvalidWorkflowStateError):
        runner.resume_workflow(workflow)


def test_runner_config_from_file() -> None:
    """Test loading configuration from a file."""
    config_data = '{"portia_api_key": "file-key", "openai_api_key": "file-openai-key"}'
    config_file = Path("config.json")
    config_file.write_text(config_data)

    config = RunnerConfig.from_file(config_file)

    assert config.portia_api_key == "file-key"
    assert config.openai_api_key == "file-openai-key"
