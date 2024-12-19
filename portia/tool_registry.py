"""A ToolRegistry represents a source of tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx
from pydantic import SecretStr

from portia.errors import ToolNotFoundError
from portia.tool import PortiaRemoteTool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import SecretStr

    from portia.tool import PortiaRemoteTool, Tool


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

    def get_tool(self, tool_name: str) -> Tool:
        """Get a tool by id."""
        if tool_name in self.tools:
            return self.tools[tool_name]
        raise ToolNotFoundError(tool_name)

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


class InMemoryToolRegistry(ToolRegistry):
    """Provides a simple in memory tool registry."""

    def __init__(self) -> None:
        """Store tools in a tool set for easy access."""
        self.tools = ToolSet([])

    @classmethod
    def from_local_tools(cls, tools: Sequence[Tool]) -> InMemoryToolRegistry:
        """Easily create a local tool registry."""
        registry = InMemoryToolRegistry()
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


class APIKeyRequiredError(Exception):
    """Raised when a given API Key is missing."""


class ToolRegistrationFailedError(Exception):
    """Raised when a tool registration fails."""


class PortiaToolRegistry(ToolRegistry):
    """Provides access to portia tools."""

    def __init__(self, api_key: SecretStr | None) -> None:
        """Store tools in a tool set for easy access."""
        if not api_key:
            raise APIKeyRequiredError
        self.api_key = api_key
        self._load_tools()

    def _load_tools(self) -> None:
        # to do load tools here
        response = httpx.get(
            url="https://holsten-37277605247.us-central1.run.app/api/v0/tools/",
            headers={
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        self.tools = response.json()

    def register_tool(self, tool: Tool) -> None:
        """Register tool in registry."""
        raise ToolRegistrationFailedError(tool)

    def get_tool(self, tool_name: str) -> PortiaRemoteTool:
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
