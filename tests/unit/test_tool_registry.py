"""tests for the ToolRegistry classes."""

import pytest

from portia.errors import ToolNotFoundError
from portia.tool import Tool
from portia.tool_registry import (
    AggregatedToolRegistry,
    InMemoryToolRegistry,
    ToolRegistry,
    ToolSet,
)
from tests.utils import AdditionTool, MockTool

MOCK_TOOL_NAME = "mock tool"
OTHER_MOCK_TOOL_NAME = "other mock tool"


def test_registry_base_classes() -> None:
    """Test registry raises."""

    class MyRegistry(ToolRegistry):
        """Override to test base."""

        def get_tools(self) -> ToolSet:
            return super().get_tools()  # type: ignore  # noqa: PGH003

        def get_tool(self, tool_name: str) -> Tool:
            return super().get_tool(tool_name)  # type: ignore  # noqa: PGH003

        def register_tool(self, tool: Tool) -> None:
            return super().register_tool(tool)  # type: ignore  # noqa: PGH003

        def match_tools(self, query: str) -> ToolSet:
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


def test_tool_set_get_tool() -> None:
    """Test the ToolSet class's get_tool method."""
    tool_set = ToolSet(tools=[MockTool(name=MOCK_TOOL_NAME)])
    tool1 = tool_set.get_tool(MOCK_TOOL_NAME)
    assert tool1.name == MOCK_TOOL_NAME

    with pytest.raises(ToolNotFoundError):
        tool_set.get_tool("tool3")


def test_local_tool_registry_register_tool() -> None:
    """Test registering tools in the InMemoryToolRegistry."""
    local_tool_registry = InMemoryToolRegistry()
    local_tool_registry.register_tool(MockTool(name=MOCK_TOOL_NAME))
    tool1 = local_tool_registry.get_tool(MOCK_TOOL_NAME)
    assert tool1.name == MOCK_TOOL_NAME

    with pytest.raises(ToolNotFoundError):
        local_tool_registry.get_tool("tool3")


def test_local_tool_registry_get_and_run() -> None:
    """Test getting and running tools in the InMemoryToolRegistry."""
    local_tool_registry = InMemoryToolRegistry()
    local_tool_registry.register_tool(MockTool(name=MOCK_TOOL_NAME))
    tool1 = local_tool_registry.get_tool(MOCK_TOOL_NAME)
    tool1.run()


def test_local_tool_registry_get_tools() -> None:
    """Test the get_tools method of InMemoryToolRegistry."""
    local_tool_registry = InMemoryToolRegistry.from_local_tools(
        [MockTool(name=MOCK_TOOL_NAME), MockTool(name=OTHER_MOCK_TOOL_NAME)],
    )
    tool_set = local_tool_registry.get_tools()
    assert len(tool_set.tools) == 2
    assert any(tool == MOCK_TOOL_NAME for tool in tool_set.tools)
    assert any(tool == OTHER_MOCK_TOOL_NAME for tool in tool_set.tools)


def test_aggregated_tool_registry_get_tool() -> None:
    """Test searching across multiple registries in AggregatedToolRegistry."""
    local_tool_registry = InMemoryToolRegistry.from_local_tools([MockTool(name=MOCK_TOOL_NAME)])
    other_tool_registry = InMemoryToolRegistry.from_local_tools(
        [MockTool(name=OTHER_MOCK_TOOL_NAME)],
    )
    aggregated_tool_registry = local_tool_registry + other_tool_registry

    tool1 = aggregated_tool_registry.get_tool(MOCK_TOOL_NAME)
    assert tool1.name == MOCK_TOOL_NAME

    with pytest.raises(ToolNotFoundError):
        aggregated_tool_registry.get_tool("tool_not_found")


def test_aggregated_tool_registry_get_tools() -> None:
    """Test getting all tools from an AggregatedToolRegistry."""
    local_tool_registry = InMemoryToolRegistry.from_local_tools([MockTool(name=MOCK_TOOL_NAME)])
    other_tool_registry = InMemoryToolRegistry.from_local_tools(
        [MockTool(name=OTHER_MOCK_TOOL_NAME)],
    )
    aggregated_tool_registry = local_tool_registry + other_tool_registry

    tool_set = aggregated_tool_registry.get_tools()
    assert len(tool_set.tools) == 2
    assert any(tool == MOCK_TOOL_NAME for tool in tool_set.tools)
