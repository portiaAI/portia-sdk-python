"""A ToolRegistry represents a source of tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field, create_model

from portia.errors import DuplicateToolError, ToolNotFoundError
from portia.tool import PortiaRemoteTool, Tool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from portia.config import Config


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
    def get_tools(self) -> list[Tool]:
        """Get all tools registered with registry."""
        raise NotImplementedError("get_tools is not implemented")

    def match_tools(self, query: str) -> list[Tool]:  # noqa: ARG002 - useful to have variable name
        """Provide a set of tools that match a given query.

        This is optional to implement and will default to provide all tools.
        """
        return self.get_tools()

    def __add__(self, other: ToolRegistry) -> ToolRegistry:
        """Return an aggregated tool registry."""
        # Ensure uniqueness of tools
        self_tools = self.get_tools()
        other_tools = other.get_tools()
        tool_names = set()
        for tool in [*self_tools, *other_tools]:
            if tool.name in tool_names:
                raise DuplicateToolError(tool.name)
            tool_names.add(tool.name)

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

    def get_tools(self) -> list[Tool]:
        """Get all tools from all registries."""
        tools = []
        for registry in self.registries:
            tools += registry.get_tools()
        return tools

    def match_tools(self, query: str) -> list[Tool]:
        """Get all tools from all registries."""
        tools = []
        for registry in self.registries:
            tools += registry.match_tools(query)
        return tools


class InMemoryToolRegistry(ToolRegistry):
    """Provides a simple in memory tool registry."""

    def __init__(self) -> None:
        """Store tools in a tool set for easy access."""
        self.tools = []

    @classmethod
    def from_local_tools(cls, tools: Sequence[Tool]) -> InMemoryToolRegistry:
        """Easily create a local tool registry."""
        registry = InMemoryToolRegistry()
        for t in tools:
            registry.register_tool(t)
        return registry

    def register_tool(self, tool: Tool) -> None:
        """Register tool in registry."""
        if self._get_tool(tool.name):
            raise DuplicateToolError(tool.name)
        self.tools.append(tool)

    def _get_tool(self, tool_name: str) -> Tool | None:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def get_tool(self, tool_name: str) -> Tool:
        """Get the tool from the registry."""
        tool = self._get_tool(tool_name)
        if not tool:
            raise ToolNotFoundError(tool_name)
        return tool

    def get_tools(self) -> list[Tool]:
        """Get all tools."""
        return self.tools


class APIKeyRequiredError(Exception):
    """Raised when a given API Key is missing."""


class ToolRegistrationFailedError(Exception):
    """Raised when a tool registration fails."""


class PortiaToolRegistry(ToolRegistry):
    """Provides access to portia tools."""

    def __init__(self, config: Config) -> None:
        """Store tools in a tool set for easy access."""
        self.api_key = config.must_get_api_key("portia_api_key")
        self.api_endpoint = config.must_get("portia_api_endpoint", str)
        self.tools = {}
        self._load_tools()

    def _generate_pydantic_model(self, model_name: str, schema: dict[str, Any]) -> type[BaseModel]:
        # Map JSON schema types to Python types
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        # Extract properties and required fields
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        # Define fields for the model
        fields = {
            key: (
                type_mapping.get(value.get("type"), Any),
                Field(default=None) if key not in required else Field(...),
            )
            for key, value in properties.items()
        }

        # Create the Pydantic model dynamically
        return create_model(model_name, **fields)  # type: ignore  # noqa: PGH003 - We want to use default config

    def _load_tools(self) -> None:
        response = httpx.get(
            url=f"{self.api_endpoint}/api/v0/tools/descriptions/",
            headers={
                "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        tools = {}
        for raw_tool in response.json():
            tool = PortiaRemoteTool(
                id=raw_tool["tool_id"],
                name=raw_tool["tool_name"],
                description=raw_tool["description"]["overview_description"],
                args_schema=self._generate_pydantic_model(
                    raw_tool["tool_name"],
                    raw_tool["schema"],
                ),
                output_schema=(
                    raw_tool["description"]["overview"],
                    raw_tool["description"]["output_description"],
                ),
                # pass API info
                api_key=self.api_key,
                api_endpoint=self.api_endpoint,
            )
            tools[raw_tool["tool_name"]] = tool
        self.tools = tools

    def register_tool(self, tool: Tool) -> None:
        """Register tool in registry."""
        raise ToolRegistrationFailedError(tool)

    def get_tool(self, tool_name: str) -> PortiaRemoteTool:
        """Get the tool from the registry."""
        if tool_name in self.tools:
            return self.tools[tool_name]

        raise ToolNotFoundError(tool_name)

    def get_tools(self) -> list[Tool]:
        """Get all tools."""
        return list(self.tools.values())
