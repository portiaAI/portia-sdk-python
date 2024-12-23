"""Portia Cloud Tests."""

import pytest
from pydantic import SecretStr

from portia.config import StorageClass, default_config
from portia.errors import ToolNotFoundError
from portia.runner import Runner
from portia.tool import ToolHardError
from portia.tool_registry import (
    InMemoryToolRegistry,
    PortiaToolRegistry,
    ToolRegistrationFailedError,
)
from portia.workflow import WorkflowState
from tests.utils import AdditionTool


def test_runner_run_query_with_cloud() -> None:
    """Test running a simple query using the Runner."""
    config = default_config()
    config.storage_class = StorageClass.CLOUD

    tool_registry = PortiaToolRegistry(config=config)
    runner = Runner(config=config, tool_registry=tool_registry)
    query = "Whats the weather in London"

    workflow = runner.run_query(query)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.final_output

    storage = runner.storage
    # check we can get items back
    storage.get_plan(workflow.plan_id)
    storage.get_workflow(
        workflow.id,
    )  # TODO: re-enable when https://github.com/portiaAI/backend-holsten/pull/67 is merged  # noqa: E501, TD002, TD003


def test_run_tool_error() -> None:
    """Test running a simple query using the Runner."""
    config = default_config()
    config.storage_class = StorageClass.CLOUD

    registry = PortiaToolRegistry(
        config=config,
    )
    with pytest.raises(ToolNotFoundError):
        registry.get_tool("Not a Tool")

    with pytest.raises(ToolRegistrationFailedError):
        registry.register_tool(AdditionTool())

    tool = registry.get_tool("Weather Tool")
    tool.api_key = SecretStr("123")
    with pytest.raises(ToolHardError):
        tool.run()


def test_runner_run_query_with_cloud_and_local() -> None:
    """Test running a simple query using the Runner."""
    config = default_config()
    config.storage_class = StorageClass.CLOUD

    registry = InMemoryToolRegistry.from_local_tools([AdditionTool()]) + PortiaToolRegistry(
        config=config,
    )

    runner = Runner(config=config, tool_registry=registry)
    query = "Get the temperature in London and Sydney and then add the two temperatures together."

    workflow = runner.run_query(query)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.final_output
