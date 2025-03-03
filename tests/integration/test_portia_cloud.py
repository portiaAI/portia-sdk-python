"""Portia Cloud Tests."""

import uuid

import pytest
from pydantic import SecretStr

from portia.clarification import ActionClarification
from portia.config import Config, StorageClass
from portia.errors import ToolNotFoundError
from portia.execution_context import execution_context
from portia.plan_run import PlanRunState
from portia.portia import Portia
from portia.storage import PortiaCloudStorage
from portia.tool import PortiaRemoteTool, Tool, ToolHardError
from portia.tool_registry import (
    InMemoryToolRegistry,
    PortiaToolRegistry,
)
from tests.utils import AdditionTool, get_test_plan_run, get_test_tool_context


def test_portia_run_query_with_cloud() -> None:
    """Test running a simple query."""
    config = Config.from_default(storage_class=StorageClass.CLOUD)
    portia = Portia(config=config)
    query = "Where is the next Olympics being hosted?"

    plan_run = portia.run(query)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output

    storage = portia.storage
    # check we can get items back
    storage.get_plan(plan_run.plan_id)
    storage.get_plan_run(plan_run.id)


def test_run_tool_error() -> None:
    """Test running a simple query."""
    config = Config.from_default(storage_class=StorageClass.CLOUD)

    registry = PortiaToolRegistry(
        config=config,
    )
    with pytest.raises(ToolNotFoundError):
        registry.get_tool("Not a Tool")

    with pytest.raises(NotImplementedError):
        registry.register_tool(AdditionTool())

    tool = registry.get_tool("portia:tavily::search")
    assert isinstance(tool, PortiaRemoteTool)
    tool.api_key = SecretStr("123")
    ctx = get_test_tool_context()
    with pytest.raises(ToolHardError):
        tool.run(ctx)


def test_portia_run_query_with_cloud_and_local() -> None:
    """Test running a simple query."""
    config = Config.from_default(storage_class=StorageClass.CLOUD)

    registry = InMemoryToolRegistry.from_local_tools([AdditionTool()]) + PortiaToolRegistry(
        config=config,
    )

    portia = Portia(config=config, tools=registry)
    query = "Get the temperature in London and Sydney and then add the two temperatures together."

    plan_run = portia.run(query)
    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output


def test_portia_run_query_with_oauth() -> None:
    """Test running a simple query."""
    portia = Portia()
    query = "Star the portiaai/portia-sdk-repo"

    with execution_context(end_user_id=str(uuid.uuid4())):
        plan_run = portia.run(query)

    assert plan_run.state == PlanRunState.NEED_CLARIFICATION
    assert len(plan_run.outputs.clarifications) == 1
    assert isinstance(plan_run.outputs.clarifications[0], ActionClarification)


def test_portia_cloud_storage() -> None:
    """Test cloud storage."""
    config = Config.from_default()
    storage = PortiaCloudStorage(config)
    (plan, plan_run) = get_test_plan_run()
    storage.save_plan(plan)
    assert storage.get_plan(plan.id) == plan
    storage.save_plan_run(plan_run)
    assert storage.get_plan_run(plan_run.id) == plan_run
    assert isinstance(storage.get_plan_runs(PlanRunState.IN_PROGRESS).results, list)


def test_default_portia_has_correct_tools() -> None:
    """Test that the default portia has the correct tools."""
    portia = Portia()
    tools = portia.tool_registry.get_tools()
    assert len(tools) > 0
    assert any(tool.id == "portia:google:gmail:search_email" for tool in tools)
    assert not any(tool.id == "portia:microsoft:outlook:draft_email" for tool in tools)


def test_portia_with_microsoft_tools() -> None:
    """Test that the default portia has the correct tools."""

    # Choose to exclude gmail rather than microsoft outlook
    def exclude_gmail_filter(tool: Tool) -> bool:
        return "gmail" not in tool.id

    registry = PortiaToolRegistry(config=Config.from_default()).filter_tools(exclude_gmail_filter)
    portia = Portia(tools=registry)
    tools = portia.tool_registry.get_tools()
    assert len(tools) > 0
    assert not any(tool.id == "portia:google:gmail:search_email" for tool in tools)
    assert any(tool.id == "portia:microsoft:outlook:draft_email" for tool in tools)
