"""tests for the ToolRegistry classes."""

import pytest

from portia.tool import Tool
from portia.tool_registry import (
    LocalToolRegistry,
    ToolNotFoundError,
    ToolSet,
)

MOCK_TOOL_NAME = "mock tool"


class MockTool(Tool):
    """A mock tool class for testing purposes."""

    id: str = "mock_tool"
    name: str = MOCK_TOOL_NAME
    description: str = "do nothing"

    def run(self) -> None:
        """Do nothing."""
        return


def test_tool_set_get_tool() -> None:
    """Test the ToolSet class's get_tool method."""
    tool_set = ToolSet(tools=[MockTool()])
    tool1 = tool_set.get_tool(MOCK_TOOL_NAME)
    assert tool1.name == MOCK_TOOL_NAME

    with pytest.raises(ToolNotFoundError):
        tool_set.get_tool("tool3")


def test_local_tool_registry_register_tool() -> None:
    """Test registering tools in the LocalToolRegistry."""
    local_tool_registry = LocalToolRegistry()
    local_tool_registry.register_tool(MockTool())
    tool1 = local_tool_registry.get_tool(MOCK_TOOL_NAME)
    assert tool1.name == MOCK_TOOL_NAME

    with pytest.raises(ToolNotFoundError):
        local_tool_registry.get_tool("tool3")


def test_local_tool_registry_get_tools() -> None:
    """Test the get_tools method of LocalToolRegistry."""
    local_tool_registry = LocalToolRegistry.from_local_tools([MockTool(), MockTool()])
    tool_set = local_tool_registry.get_tools()
    assert len(tool_set.tools) == 1
    assert any(tool.name == MOCK_TOOL_NAME for tool in tool_set.tools)


def test_aggregated_tool_registry_get_tool() -> None:
    """Test searching across multiple registries in AggregatedToolRegistry."""
    local_tool_registry = LocalToolRegistry.from_local_tools([MockTool()])
    other_tool_registry = LocalToolRegistry.from_local_tools([MockTool()])
    aggregated_tool_registry = local_tool_registry + other_tool_registry

    tool1 = aggregated_tool_registry.get_tool(MOCK_TOOL_NAME)
    assert tool1.name == MOCK_TOOL_NAME

    with pytest.raises(ToolNotFoundError):
        aggregated_tool_registry.get_tool("tool_not_found")


def test_aggregated_tool_registry_get_tools() -> None:
    """Test getting all tools from an AggregatedToolRegistry."""
    local_tool_registry = LocalToolRegistry.from_local_tools([MockTool()])
    other_tool_registry = LocalToolRegistry.from_local_tools([MockTool()])
    aggregated_tool_registry = local_tool_registry + other_tool_registry

    tool_set = aggregated_tool_registry.get_tools()
    assert len(tool_set.tools) == 2
    assert any(tool.name == MOCK_TOOL_NAME for tool in tool_set.tools)
