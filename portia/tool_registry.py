"""A ToolRegistry represents a source of tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from portia.errors import ToolNotFoundError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from portia.tool import Tool


class ToolSet:
    """ToolSet is a convenience type for a set of Tools."""

    def __init__(self, tools: list[Tool]) -> None:
        """Initialize a set of tools."""
        self.tools: dict[str, Tool] = {}
        for tool in tools:
            self.tools[tool.name] = tool

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the set."""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool:
        """Get a tool by name."""
        if name in self.tools:
            return self.tools[name]
        raise ToolNotFoundError(name)

    def get_tools(self) -> list[Tool]:
        """Get all tools."""
        return list(self.tools.values())

    def __add__(self, other: ToolSet) -> ToolSet:
        """Return an aggregated tool set."""
        new_tools = list(self.tools.values()) + list(other.tools.values())
        return ToolSet(new_tools)


class ToolRegistry(ABC):
    """ToolRegistry is the base interface for managing tools."""

    @abstractmethod
    def register_tool(self, tool: Tool) -> None:
        """Register a new tool."""
        raise NotImplementedError("register_tool is not implemented")

    @abstractmethod
    def get_tool(self, tool_name: str) -> Tool:
        """Retrieve a tool's information."""
        raise NotImplementedError("get_tool is not implemented")

    @abstractmethod
    def get_tools(self) -> ToolSet:
        """Get all tools registered with registry."""
        raise NotImplementedError("get_tools is not implemented")

    def match_tools(self, query: str) -> ToolSet:  # noqa: ARG002 - useful to have variable name
        """Provide a set of tools that match a given query.

        This is optional to implement and will default to provide all tools.
        """
        return self.get_tools()

    def __add__(self, other: ToolRegistry) -> ToolRegistry:
        """Return an aggregated tool registry."""
        return AggregatedToolRegistry([self, other])


class AggregatedToolRegistry(ToolRegistry):
    """An interface over a set of tool registries."""

    def __init__(self, registries: list[ToolRegistry]) -> None:
        """Set the registries we will use."""
        self.registries = registries

    def register_tool(self, tool: Tool) -> None:
        """Tool registration should happen in individual registries."""
        raise NotImplementedError("tool registration should happen in individual registries.")

    def get_tool(self, tool_name: str) -> Tool:
        """Search across all registries for a given tool, returning first match."""
        for registry in self.registries:
            try:
                return registry.get_tool(tool_name)
            except ToolNotFoundError:  # noqa: PERF203
                continue
        raise ToolNotFoundError(tool_name)

    def get_tools(self) -> ToolSet:
        """Get all tools from all registries."""
        tools = ToolSet([])
        for registry in self.registries:
            tools += registry.get_tools()
        return tools

    def match_tools(self, query: str) -> ToolSet:
        """Get all tools from all registries."""
        tools = ToolSet([])
        for registry in self.registries:
            tools += registry.match_tools(query)
        return tools


class LocalToolRegistry(ToolRegistry):
    """Provides a simple in memory tool registry."""

    def __init__(self) -> None:
        """Store tools in a tool set for easy access."""
        self.tools = ToolSet([])

    @classmethod
    def from_local_tools(cls, tools: Sequence[Tool]) -> LocalToolRegistry:
        """Easily create a local tool registry."""
        registry = LocalToolRegistry()
        for t in tools:
            registry.register_tool(t)
        return registry

    def register_tool(self, tool: Tool) -> None:
        """Register tool in registry."""
        self.tools.add_tool(tool)

    def get_tool(self, tool_name: str) -> Tool:
        """Get the tool from the registry."""
        tool = self.tools.get_tool(
            tool_name,
        )
        if not tool:
            raise ToolNotFoundError(tool_name)
        return tool

    def get_tools(self) -> ToolSet:
        """Get all tools."""
        return self.tools
