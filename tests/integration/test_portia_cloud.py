"""E2E Tests."""

from portia.config import StorageClass, default_config
from portia.runner import Runner
from portia.tool_registry import InMemoryToolRegistry, PortiaToolRegistry
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
