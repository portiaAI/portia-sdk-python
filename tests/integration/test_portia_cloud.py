"""Portia Cloud Tests."""

import uuid

import pytest
from pydantic import SecretStr

from portia.clarification import ActionClarification
from portia.config import Config, StorageClass
from portia.errors import ToolNotFoundError
from portia.execution_context import execution_context, get_execution_context
from portia.runner import Runner
from portia.storage import PortiaCloudStorage
from portia.tool import ToolHardError
from portia.tool_registry import (
    InMemoryToolRegistry,
    PortiaToolRegistry,
)
from portia.workflow import WorkflowState
from tests.utils import AdditionTool, get_test_workflow


def test_runner_run_query_with_cloud() -> None:
    """Test running a simple query using the Runner."""
    config = Config.from_default(storage_class=StorageClass.CLOUD)
    tool_registry = PortiaToolRegistry(config=config)
    runner = Runner(config=config, tool_registry=tool_registry)
    query = "Where is the next Olympics being hosted?"

    workflow = runner.execute_query(query)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.outputs.final_output

    storage = runner.storage
    # check we can get items back
    storage.get_plan(workflow.plan_id)
    storage.get_workflow(workflow.id)


def test_run_tool_error() -> None:
    """Test running a simple query using the Runner."""
    config = Config.from_default(storage_class=StorageClass.CLOUD)

    registry = PortiaToolRegistry(
        config=config,
    )
    with pytest.raises(ToolNotFoundError):
        registry.get_tool("Not a Tool")

    with pytest.raises(NotImplementedError):
        registry.register_tool(AdditionTool())

    tool = registry.get_tool("portia::search_tool")
    tool.api_key = SecretStr("123")
    ctx = get_execution_context()
    with pytest.raises(ToolHardError):
        tool.run(ctx)


def test_runner_run_query_with_cloud_and_local() -> None:
    """Test running a simple query using the Runner."""
    config = Config.from_default(storage_class=StorageClass.CLOUD)

    registry = InMemoryToolRegistry.from_local_tools([AdditionTool()]) + PortiaToolRegistry(
        config=config,
    )

    runner = Runner(config=config, tool_registry=registry)
    query = "Get the temperature in London and Sydney and then add the two temperatures together."

    workflow = runner.execute_query(query)
    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.outputs.final_output


def test_runner_run_query_with_oauth() -> None:
    """Test running a simple query using the Runner."""
    config = Config.from_default()
    tool_registry = PortiaToolRegistry(config=config)
    runner = Runner(config=config, tool_registry=tool_registry)
    query = "Star the portiaai/portia-sdk-repo"

    with execution_context(end_user_id=str(uuid.uuid4())):
        workflow = runner.execute_query(query)

    assert workflow.state == WorkflowState.NEED_CLARIFICATION
    assert len(workflow.outputs.clarifications) == 1
    assert isinstance(workflow.outputs.clarifications[0], ActionClarification)


def test_portia_cloud_storage() -> None:
    """Test disk storage."""
    config = Config.from_default()
    storage = PortiaCloudStorage(config)
    (plan, workflow) = get_test_workflow()
    storage.save_plan(plan)
    assert storage.get_plan(plan.id) == plan
    storage.save_workflow(workflow)
    assert storage.get_workflow(workflow.id) == workflow
    assert isinstance(storage.get_workflows(WorkflowState.IN_PROGRESS), list)
