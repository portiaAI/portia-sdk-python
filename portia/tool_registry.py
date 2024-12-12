"""A ToolRegistry represents a source of tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from portia.tool import Tool


class ToolNotFoundError(Exception):
    """Custom error class when tools aren't found."""


class ToolSet:
    """ToolSet is a convenience type for a set of Tools."""

    def __init__(self, tools: list[Tool]) -> None:
        """Initialize a set of tools."""
        self.tools = tools

    def get_tool(self, name: str) -> Tool:
        """Get a tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise ToolNotFoundError


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
        raise ToolNotFoundError

    def get_tools(self) -> ToolSet:
        """Get all tools from all registries."""
        tools: list[Tool] = []
        for registry in self.registries:
            tools.extend(registry.get_tools().tools)
        return ToolSet(tools)

    def match_tools(self, query: str) -> ToolSet:
        """Get all tools from all registries."""
        tools: list[Tool] = []
        for registry in self.registries:
            tools.extend(registry.match_tools(query).tools)
        return ToolSet(tools)


class LocalToolRegistry(ToolRegistry):
    """Provides a simple in memory tool registry."""

    def __init__(self) -> None:
        """Store tools in a dict for easy access."""
        self.registry: dict[str, Tool] = {}

    @classmethod
    def from_local_tools(cls, tools: Sequence[Tool]) -> LocalToolRegistry:
        """Easily create a local tool registry."""
        registry = LocalToolRegistry()
        for t in tools:
            registry.register_tool(t)
        return registry

    def register_tool(self, tool: Tool) -> None:
        """Register tool in registry."""
        self.registry[tool.name] = tool

    def get_tool(self, tool_name: str) -> Tool:
        """Get the tool from the registry."""
        tool = self.registry.get(
            tool_name,
        )
        if not tool:
            raise ToolNotFoundError
        return tool

    def get_tools(self) -> ToolSet:
        """Get all tools."""
        return ToolSet(list(self.registry.values()))
