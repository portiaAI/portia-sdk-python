"""tests for the ToolRegistry classes."""

import pytest

from portia.errors import DuplicateToolError, ToolNotFoundError
from portia.execution_context import get_execution_context
from portia.tool import Tool
from portia.tool_registry import (
    AggregatedToolRegistry,
    InMemoryToolRegistry,
    ToolRegistry,
)
from tests.utils import AdditionTool, MockTool

MOCK_TOOL_ID = "mock_tool"
OTHER_MOCK_TOOL_ID = "other_mock_tool"


def test_registry_base_classes() -> None:
    """Test registry raises."""

    class MyRegistry(ToolRegistry):
        """Override to test base."""

        def get_tools(self) -> list[Tool]:
            return super().get_tools()  # type: ignore  # noqa: PGH003

        def get_tool(self, tool_id: str) -> Tool:
            return super().get_tool(tool_id)  # type: ignore  # noqa: PGH003

        def register_tool(self, tool: Tool) -> None:
            return super().register_tool(tool)  # type: ignore  # noqa: PGH003

        def match_tools(self, query: str) -> list[Tool]:
            return super().match_tools(query)

    registry = MyRegistry()

    with pytest.raises(NotImplementedError):
        registry.get_tools()

    with pytest.raises(NotImplementedError):
        registry.get_tool("1")

    with pytest.raises(NotImplementedError):
        registry.register_tool(AdditionTool())

    with pytest.raises(NotImplementedError):
        registry.match_tools("match")

    agg_registry = AggregatedToolRegistry(registries=[registry])
    with pytest.raises(NotImplementedError):
        agg_registry.register_tool(AdditionTool())


def test_local_tool_registry_register_tool() -> None:
    """Test registering tools in the InMemoryToolRegistry."""
    local_tool_registry = InMemoryToolRegistry()
    local_tool_registry.register_tool(MockTool(id=MOCK_TOOL_ID))
    tool1 = local_tool_registry.get_tool(MOCK_TOOL_ID)
    assert tool1.id == MOCK_TOOL_ID

    with pytest.raises(ToolNotFoundError):
        local_tool_registry.get_tool("tool3")

    with pytest.raises(DuplicateToolError):
        local_tool_registry.register_tool(MockTool(id=MOCK_TOOL_ID))


def test_local_tool_registry_get_and_run() -> None:
    """Test getting and running tools in the InMemoryToolRegistry."""
    local_tool_registry = InMemoryToolRegistry()
    local_tool_registry.register_tool(MockTool(id=MOCK_TOOL_ID))
    tool1 = local_tool_registry.get_tool(MOCK_TOOL_ID)
    ctx = get_execution_context()
    tool1.run(ctx)


def test_local_tool_registry_get_tools() -> None:
    """Test the get_tools method of InMemoryToolRegistry."""
    local_tool_registry = InMemoryToolRegistry.from_local_tools(
        [MockTool(id=MOCK_TOOL_ID), MockTool(id=OTHER_MOCK_TOOL_ID)],
    )
    tools = local_tool_registry.get_tools()
    assert len(tools) == 2
    assert any(tool.id == MOCK_TOOL_ID for tool in tools)
    assert any(tool.id == OTHER_MOCK_TOOL_ID for tool in tools)


def test_aggregated_tool_registry_duplicate_tool() -> None:
    """Test searching across multiple registries in AggregatedToolRegistry."""
    local_tool_registry = InMemoryToolRegistry.from_local_tools([MockTool(id=MOCK_TOOL_ID)])
    other_tool_registry = InMemoryToolRegistry.from_local_tools(
        [MockTool(id=MOCK_TOOL_ID)],
    )
    with pytest.raises(DuplicateToolError):
        aggregated_tool_registry = local_tool_registry + other_tool_registry  # noqa: F841


def test_aggregated_tool_registry_get_tool() -> None:
    """Test searching across multiple registries in AggregatedToolRegistry."""
    local_tool_registry = InMemoryToolRegistry.from_local_tools([MockTool(id=MOCK_TOOL_ID)])
    other_tool_registry = InMemoryToolRegistry.from_local_tools(
        [MockTool(id=OTHER_MOCK_TOOL_ID)],
    )
    aggregated_tool_registry = local_tool_registry + other_tool_registry

    tool1 = aggregated_tool_registry.get_tool(MOCK_TOOL_ID)
    assert tool1.id == MOCK_TOOL_ID

    with pytest.raises(ToolNotFoundError):
        aggregated_tool_registry.get_tool("tool_not_found")


def test_aggregated_tool_registry_get_tools() -> None:
    """Test getting all tools from an AggregatedToolRegistry."""
    local_tool_registry = InMemoryToolRegistry.from_local_tools([MockTool(id=MOCK_TOOL_ID)])
    other_tool_registry = InMemoryToolRegistry.from_local_tools(
        [MockTool(id=OTHER_MOCK_TOOL_ID)],
    )
    aggregated_tool_registry = local_tool_registry + other_tool_registry

    tools = aggregated_tool_registry.get_tools()
    assert len(tools) == 2
    assert any(tool.id == MOCK_TOOL_ID for tool in tools)
