"""A ToolRegistry represents a source of tools.

This module defines various implementations of `ToolRegistry`, which is responsible for managing
and interacting with tools. It provides interfaces for registering, retrieving, and listing tools.
The `ToolRegistry` can also support aggregation of multiple registries and searching for tools
based on queries.

Classes:
    ToolRegistry: The base interface for managing tools.
    AggregatedToolRegistry: A registry that aggregates multiple tool registries.
    InMemoryToolRegistry: A simple in-memory implementation of `ToolRegistry`.
    PortiaToolRegistry: A tool registry that interacts with the Portia API to manage tools.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

import httpx
from pydantic import BaseModel, Field, create_model

from portia.errors import DuplicateToolError, ToolNotFoundError
from portia.logger import logger
from portia.tool import PortiaRemoteTool, Tool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from portia.config import Config


class ToolRegistry(ABC):
    """ToolRegistry is the base interface for managing tools.

    This class defines the essential methods for interacting with tool registries, including
    registering, retrieving, and listing tools. Specific tool registries should implement these
    methods.

    Methods:
        register_tool(tool: Tool) -> None:
            Registers a new tool in the registry.
        get_tool(tool_id: str) -> Tool:
            Retrieves a tool by its ID.
        get_tools() -> list[Tool]:
            Retrieves all tools in the registry.
        match_tools(query: str | None = None, tool_ids: list[str] | None = None) -> list[Tool]:
            Optionally, retrieve tools that match a given query and tool_ids. Useful to implement
            tool filtering.

    """

    @abstractmethod
    def register_tool(self, tool: Tool) -> None:
        """Register a new tool.

        Args:
            tool (Tool): The tool to be registered.

        """
        raise NotImplementedError("register_tool is not implemented")

    @abstractmethod
    def get_tool(self, tool_id: str) -> Tool:
        """Retrieve a tool's information.

        Args:
            tool_id (str): The ID of the tool to retrieve.

        Returns:
            Tool: The requested tool.

        Raises:
            ToolNotFoundError: If the tool with the given ID does not exist.

        """
        raise NotImplementedError("get_tool is not implemented")

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """Get all tools registered with the registry.

        Returns:
            list[Tool]: A list of all tools in the registry.

        """
        raise NotImplementedError("get_tools is not implemented")

    def match_tools(
        self,
        query: str | None = None,  # noqa: ARG002 - useful to have variable name
        tool_ids: list[str] | None = None,
    ) -> list[Tool]:
        """Provide a set of tools that match a given query and tool_ids.

        Args:
            query (str | None): The query to match tools against.
            tool_ids (list[str] | None): The list of tool ids to match.

        Returns:
            list[Tool]: A list of tools matching the query.

        This method is useful to implement tool filtering whereby only a selection of tools are
        passed to the Planner based on the query.
        This method is optional to implement and will default to providing all tools.

        """
        return (
            [tool for tool in self.get_tools() if tool.id in tool_ids]
            if tool_ids
            else self.get_tools()
        )

    @abstractmethod
    def filter_tools(
        self,
        filter_func: Callable[[Tool], bool],
    ) -> ToolRegistry:
        """Return a new registry with the tools filtered by the filter function.

        Args:
            filter_func (Callable[[Tool], bool]): A filter to select a subset of tools (return true
            to include a tool)

        """
        raise NotImplementedError("filter_tools is not implemented")

    def __add__(self, other: ToolRegistry | list[Tool]) -> ToolRegistry:
        """Return an aggregated tool registry combining two registries or a registry and tool list.

        Tool IDs must be unique across the two registries otherwise an error will be thrown.

        Args:
            other (ToolRegistry): Another tool registry to be combined.

        Returns:
            AggregatedToolRegistry: A new tool registry containing tools from both registries.

        """
        return self._add(other)

    def __radd__(self, other: ToolRegistry | list[Tool]) -> ToolRegistry:
        """Return an aggregated tool registry combining two registries or a registry and tool list.

        Tool IDs must be unique across the two registries otherwise an error will be thrown.

        Args:
            other (ToolRegistry): Another tool registry to be combined.

        Returns:
            AggregatedToolRegistry: A new tool registry containing tools from both registries.

        """
        return self._add(other)

    def _add(self, other: ToolRegistry | list[Tool]) -> AggregatedToolRegistry:
        """Add a tool registry or Tool list to the current registry."""
        other_registry = (
            other
            if isinstance(other, ToolRegistry)
            else InMemoryToolRegistry.from_local_tools(other)
        )
        self_tools = self.get_tools()
        other_tools = other_registry.get_tools()
        tool_ids = set()
        for tool in [*self_tools, *other_tools]:
            if tool.id in tool_ids:
                logger().warning(
                    f"Duplicate tool ID found: {tool.id}. Unintended behavior may occur.",
                )
            tool_ids.add(tool.id)

        return AggregatedToolRegistry([self, other_registry])


class AggregatedToolRegistry(ToolRegistry):
    """An interface over a set of tool registries.

    This class aggregates multiple tool registries, allowing the user to retrieve tools from
    any of the registries in the collection.
    """

    def __init__(self, registries: list[ToolRegistry]) -> None:
        """Initialize the aggregated tool registry with a list of registries.

        Args:
            registries (list[ToolRegistry]): A list of tool registries to aggregate.

        """
        self.registries = registries

    def register_tool(self, tool: Tool) -> None:
        """Throw not implemented error as registration should happen in individual registries."""
        raise NotImplementedError("tool registration should happen in individual registries.")

    def get_tool(self, tool_id: str) -> Tool:
        """Search across all registries for a given tool, returning the first match.

        Args:
            tool_id (str): The ID of the tool to retrieve.

        Returns:
            Tool: The requested tool.

        Raises:
            ToolNotFoundError: If the tool with the given ID does not exist in any registry.

        """
        for registry in self.registries:
            try:
                return registry.get_tool(tool_id)
            except ToolNotFoundError:  # noqa: PERF203
                continue
        raise ToolNotFoundError(tool_id)

    def get_tools(self) -> list[Tool]:
        """Get all tools from all registries.

        Returns:
            list[Tool]: A list of all tools across all registries.

        """
        tools = []
        for registry in self.registries:
            tools += registry.get_tools()
        return tools

    def match_tools(
        self,
        query: str | None = None,
        tool_ids: list[str] | None = None,
    ) -> list[Tool]:
        """Get all tools from all registries that match the query and tool_ids.

        Args:
            query (str | None): The query to match tools against.
            tool_ids (list[str] | None): The list of tool ids to match.

        Returns:
            list[Tool]: A list of tools matching the query from all registries.

        """
        tools = []
        for registry in self.registries:
            tools += registry.match_tools(query, tool_ids)
        return tools

    def filter_tools(
        self,
        filter_func: Callable[[Tool], bool],
    ) -> ToolRegistry:
        """Return a new registry with the tools filtered by the filter function."""
        registries = [registry.filter_tools(filter_func) for registry in self.registries]
        return AggregatedToolRegistry(registries)


class InMemoryToolRegistry(ToolRegistry):
    """Provides a simple in-memory tool registry.

    This class stores tools in memory, allowing for quick access without persistence.
    """

    def __init__(self) -> None:
        """Initialize the registry with an empty list of tools."""
        self.tools = []

    @classmethod
    def from_local_tools(cls, tools: Sequence[Tool]) -> InMemoryToolRegistry:
        """Easily create a local tool registry from a sequence of tools.

        Args:
            tools (Sequence[Tool]): A sequence of tools to initialize the registry.

        Returns:
            InMemoryToolRegistry: A new in-memory tool registry.

        """
        registry = InMemoryToolRegistry()
        for t in tools:
            registry.register_tool(t)
        return registry

    def register_tool(self, tool: Tool) -> None:
        """Register tool in the in-memory registry.

        Args:
            tool (Tool): The tool to register.

        Raises:
            DuplicateToolError: If the tool ID already exists in the registry.

        """
        if self._get_tool(tool.id):
            raise DuplicateToolError(tool.id)
        self.tools.append(tool)

    def _get_tool(self, tool_id: str) -> Tool | None:
        """Retrieve a tool by ID."""
        for tool in self.tools:
            if tool.id == tool_id:
                return tool
        return None

    def get_tool(self, tool_id: str) -> Tool:
        """Get the tool from the in-memory registry.

        Args:
            tool_id (str): The ID of the tool to retrieve.

        Returns:
            Tool: The requested tool.

        Raises:
            ToolNotFoundError: If the tool with the given ID does not exist.

        """
        tool = self._get_tool(tool_id)
        if not tool:
            raise ToolNotFoundError(tool_id)
        return tool

    def get_tools(self) -> list[Tool]:
        """Get all tools in the in-memory registry.

        Returns:
            list[Tool]: A list of all tools in the registry.

        """
        return self.tools

    def filter_tools(
        self,
        filter_func: Callable[[Tool], bool],
    ) -> ToolRegistry:
        """Return a new registry with the tools filtered by the filter function."""
        return InMemoryToolRegistry.from_local_tools(
            [tool for tool in self.tools if filter_func(tool)],
        )


EXCLUDED_BY_DEFAULT_TOOL_REGEXS: frozenset[str] = frozenset(
    {
        # Exclude Outlook by default as it clashes with Gmail
        "portia::microsoft_outlook::*",
    },
)


def get_default_tool_registry(config: Config) -> ToolRegistry:
    """Get the default tool registry based on the configuration."""

    def default_tool_filter(tool: Tool) -> bool:
        """Filter to get the default set of tools offered by Portia cloud."""
        return not any(re.match(regex, tool.id) for regex in EXCLUDED_BY_DEFAULT_TOOL_REGEXS)

    return PortiaToolRegistry(config).filter_tools(default_tool_filter)


class PortiaToolRegistry(ToolRegistry):
    """Provides access to Portia tools.

    This class interacts with the Portia API to retrieve and manage tools.
    """

    def __init__(
        self,
        config: Config,
        tools: dict[str, Tool] | None = None,
    ) -> None:
        """Initialize the PortiaToolRegistry with the given configuration.

        Args:
            config (Config): The configuration containing the API key and endpoint.
            tools (list[Tool] | None): A list of tools to create the registry with.
              If not provided, all tools will be loaded from the Portia API.

        """
        self.api_key = config.must_get_api_key("portia_api_key")
        self.config = config
        self.api_endpoint = config.must_get("portia_api_endpoint", str)
        self.tools = tools or self._load_tools()

    def _generate_pydantic_model(self, model_name: str, schema: dict[str, Any]) -> type[BaseModel]:
        """Generate a Pydantic model based on a JSON schema.

        Args:
            model_name (str): The name of the Pydantic model.
            schema (dict[str, Any]): The schema to generate the model from.

        Returns:
            type[BaseModel]: The generated Pydantic model class.

        """
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
                Field(
                    default=None,
                    description=value.get("description", ""),
                )
                if key not in required
                else Field(
                    ...,
                    description=value.get("description", ""),
                ),
            )
            for key, value in properties.items()
        }

        # Create the Pydantic model dynamically
        return create_model(model_name, **fields)  # type: ignore  # noqa: PGH003 - We want to use default config

    def _load_tools(self) -> list[Tool]:
        """Load the tools from the API into the into the internal storage."""
        response = httpx.get(
            url=f"{self.api_endpoint}/api/v0/tools/descriptions/",
            headers={
                "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        response.raise_for_status()
        tools = {}
        for raw_tool in response.json():
            tool = PortiaRemoteTool(
                id=raw_tool["tool_id"],
                name=raw_tool["tool_name"],
                should_summarize=raw_tool.get("should_summarize", False),
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
            tools[raw_tool["tool_id"]] = tool
        return tools

    def register_tool(self, tool: Tool) -> None:
        """Throw not implemented error as registration can't be done in this registry."""
        raise NotImplementedError("Cannot register tools in the PortiaToolRegistry")

    def get_tool(self, tool_id: str) -> PortiaRemoteTool:
        """Get the tool from the tool set.

        Args:
            tool_id (str): The ID of the tool to retrieve.

        Returns:
            Tool: The requested tool.

        Raises:
            ToolNotFoundError: If the tool with the given ID does not exist.

        """
        if tool_id in self.tools:
            return self.tools[tool_id]

        raise ToolNotFoundError(tool_id)

    def get_tools(self) -> list[Tool]:
        """Get all tools in the registry.

        Returns:
            list[Tool]: A list of all tools in the registry.

        """
        return list(self.tools.values())

    def filter_tools(
        self,
        filter_func: Callable[[Tool], bool],
    ) -> ToolRegistry:
        """Return a new registry with the tools filtered by the filter function."""
        return PortiaToolRegistry(
            self.config,
            {tool.id: tool for tool in self.get_tools() if filter_func(tool)},
        )
