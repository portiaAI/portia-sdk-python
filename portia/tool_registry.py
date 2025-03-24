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
    MCPToolRegistry: A tool registry that interacts with a locally running MCP server.
"""

from __future__ import annotations

import asyncio
import os
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Literal, Union

from pydantic import BaseModel, Field, create_model

from portia.cloud import PortiaCloudClient
from portia.errors import DuplicateToolError, ToolNotFoundError
from portia.logger import logger
from portia.mcp_session import (
    McpClientConfig,
    SseMcpClientConfig,
    StdioMcpClientConfig,
    get_mcp_session,
)
from portia.open_source_tools.calculator_tool import CalculatorTool
from portia.open_source_tools.image_understanding_tool import ImageUnderstandingTool
from portia.open_source_tools.llm_tool import LLMTool
from portia.open_source_tools.local_file_reader_tool import FileReaderTool
from portia.open_source_tools.local_file_writer_tool import FileWriterTool
from portia.open_source_tools.search_tool import SearchTool
from portia.open_source_tools.weather import WeatherTool
from portia.tool import PortiaMcpTool, PortiaRemoteTool, Tool

if TYPE_CHECKING:
    from collections.abc import Sequence

    import mcp

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
        passed to the PlanningAgent based on the query.
        This method is optional to implement and will default to providing all tools.

        """
        return (
            [tool for tool in self.get_tools() if tool.id in tool_ids]
            if tool_ids
            else self.get_tools()
        )

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
                    f"Duplicate tool ID found: {tool.id!s}. Unintended behavior may occur.",
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


class PortiaToolRegistry(ToolRegistry):
    """Provides access to Portia tools.

    This class interacts with the Portia API to retrieve and manage tools.
    """

    EXCLUDED_BY_DEFAULT_TOOL_REGEXS: frozenset[str] = frozenset(
        {
            # Exclude Outlook by default as it clashes with Gmail
            "portia:microsoft:outlook:*",
        },
    )

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
        self.config = config
        self.client = PortiaCloudClient().get_client(config)
        self.tools = tools or self._load_tools()

    @classmethod
    def with_default_tool_filter(cls, config: Config) -> ToolRegistry:
        """Create a PortiaToolRegistry with a default tool filter."""
        def default_tool_filter(tool: Tool) -> bool:
            """Filter to get the default set of tools offered by Portia cloud."""
            return not any(
                re.match(regex, tool.id)
                for regex
                in cls.EXCLUDED_BY_DEFAULT_TOOL_REGEXS
            )
        return PortiaToolRegistry(config).filter_tools(default_tool_filter)

    @classmethod
    def with_unauthenticated_client(cls, config: Config) -> ToolRegistry:
        """Create a PortiaToolRegistry with an unauthenticated client."""
        registry = PortiaToolRegistry.with_default_tool_filter(config)
        if isinstance(registry, PortiaToolRegistry):
            registry.client = PortiaCloudClient.new_client(config, allow_unauthenticated=True)
        return registry

    def _load_tools(self) -> dict[str, Tool]:
        """Load the tools from the API into the into the internal storage."""
        response = self.client.get(
            url="/api/v0/tools/descriptions/",
        )
        response.raise_for_status()
        tools = {}
        for raw_tool in response.json():
            tool = PortiaRemoteTool(
                id=raw_tool["tool_id"],
                name=raw_tool["tool_name"],
                should_summarize=raw_tool.get("should_summarize", False),
                description=raw_tool["description"]["overview_description"],
                args_schema=generate_pydantic_model_from_json_schema(
                    raw_tool["tool_name"],
                    raw_tool["schema"],
                ),
                output_schema=(
                    raw_tool["description"]["overview"],
                    raw_tool["description"]["output_description"],
                ),
                # pass API info
                client=self.client,
            )
            tools[raw_tool["tool_id"]] = tool
        return tools

    def register_tool(self, tool: Tool) -> None:
        """Throw not implemented error as registration can't be done in this registry."""
        raise NotImplementedError("Cannot register tools in the PortiaToolRegistry")

    def get_tool(self, tool_id: str) -> Tool:
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


class McpToolRegistry(ToolRegistry):
    """Provides access to tools within a Model Context Protocol (MCP) server.

    See https://modelcontextprotocol.io/introduction for more information on MCP.
    """

    def __init__(self, mcp_client_config: McpClientConfig) -> None:
        """Initialize the MCPToolRegistry with the given configuration."""
        self.tools = {t.id: t for t in self._load_tools(mcp_client_config)}

    @classmethod
    def from_sse_connection(
        cls,
        server_name: str,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
    ) -> McpToolRegistry:
        """Create a new MCPToolRegistry using an SSE connection."""
        return cls(
            SseMcpClientConfig(
                server_name=server_name,
                url=url,
                headers=headers,
                timeout=timeout,
                sse_read_timeout=sse_read_timeout,
            ),
        )

    @classmethod
    def from_stdio_connection(  # noqa: PLR0913
        cls,
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        encoding: str = "utf-8",
        encoding_error_handler: Literal["strict", "ignore", "replace"] = "strict",
    ) -> McpToolRegistry:
        """Create a new MCPToolRegistry using a stdio connection."""
        return cls(
            StdioMcpClientConfig(
                server_name=server_name,
                command=command,
                args=args if args is not None else [],
                env=env,
                encoding=encoding,
                encoding_error_handler=encoding_error_handler,
            ),
        )

    def _load_tools(self, mcp_client_config: McpClientConfig) -> list[PortiaMcpTool]:
        """Get a list of tools from an MCP server wrapped at Portia tools.

        Args:
            mcp_client_config: The configuration for the MCP client

        Returns:
            A list of Portia tools

        """

        async def async_inner() -> list[PortiaMcpTool]:
            async with get_mcp_session(mcp_client_config) as session:
                logger().debug("Fetching tools from MCP server")
                tools = await session.list_tools()
                logger().debug(f"Got {len(tools.tools)} tools from MCP server")
                return [
                    self._portia_tool_from_mcp_tool(tool, mcp_client_config) for tool in tools.tools
                ]

        return asyncio.run(async_inner())

    def _portia_tool_from_mcp_tool(
        self,
        mcp_tool: mcp.Tool,
        mcp_client_config: McpClientConfig,
    ) -> PortiaMcpTool:
        """Conversion of a remote MCP server tool to a Portia tool."""
        tool_name_snake_case = re.sub(r"[^a-zA-Z0-9]+", "_", mcp_tool.name)

        description = (
            mcp_tool.description
            if mcp_tool.description is not None
            else f"{mcp_tool.name} tool from {mcp_client_config.server_name}"
        )

        return PortiaMcpTool(
            id=f"mcp:{mcp_client_config.server_name}:{tool_name_snake_case}",
            name=mcp_tool.name,
            description=description,
            args_schema=generate_pydantic_model_from_json_schema(
                f"{tool_name_snake_case}_schema",
                mcp_tool.inputSchema,
            ),
            output_schema=("str", "The response from the tool formatted as a JSON string"),
            mcp_client_config=mcp_client_config,
        )

    def register_tool(self, tool: Tool) -> None:
        """Register a new tool.

        Args:
            tool (Tool): The tool to be registered.

        """
        raise NotImplementedError("register_tool is not supported for MCPToolRegistry")

    def get_tool(self, tool_id: str) -> Tool:
        """Retrieve a tool's information.

        Args:
            tool_id (str): The ID of the tool to retrieve.

        Returns:
            Tool: The requested tool.

        Raises:
            ToolNotFoundError: If the tool with the given ID does not exist.

        """
        tool = self.tools.get(tool_id)
        if not tool:
            raise ToolNotFoundError(tool_id)
        return tool

    def get_tools(self) -> list[Tool]:
        """Get all tools registered with the registry.

        Returns:
            list[Tool]: A list of all tools in the registry.

        """
        return list(self.tools.values())


class DefaultToolRegistry(AggregatedToolRegistry):
    """A registry providing a default set of tools.

    This includes the following tools:
    - All open source tools that don't require API keys
    - Search tool if you have a Tavily API key
    - Weather tool if you have an OpenWeatherMap API key
    - Portia cloud tools if you have a Portia cloud API key
    """

    def __init__(self, config: Config) -> None:
        """Initialize the default tool registry with the given configuration."""
        in_memory_registry = InMemoryToolRegistry.from_local_tools(
            [
                CalculatorTool(),
                LLMTool(),
                FileWriterTool(),
                FileReaderTool(),
                ImageUnderstandingTool(),
            ],
        )
        if os.getenv("TAVILY_API_KEY"):
            in_memory_registry.register_tool(SearchTool())
        if os.getenv("OPENWEATHERMAP_API_KEY"):
            in_memory_registry.register_tool(WeatherTool())

        registries: list[ToolRegistry] = [in_memory_registry]
        if config.portia_api_key:
            registries.append(PortiaToolRegistry.with_default_tool_filter(config))

        super().__init__(registries)


def generate_pydantic_model_from_json_schema(
    model_name: str,
    json_schema: dict[str, Any],
) -> type[BaseModel]:
    """Generate a Pydantic model based on a JSON schema.

    Args:
        model_name (str): The name of the Pydantic model.
        json_schema (dict[str, Any]): The schema to generate the model from.

    Returns:
        type[BaseModel]: The generated Pydantic model class.

    """
    # Extract properties and required fields
    properties = json_schema.get("properties", {})
    required = set(json_schema.get("required", []))

    # Define fields for the model
    fields = dict(
        [
            _generate_field(key, value, required=key in required)
            for key, value in properties.items()
        ],
    )

    # Create the Pydantic model dynamically
    return create_model(model_name, **fields)  # type: ignore  # noqa: PGH003 - We want to use default config


def _generate_field(
    field_name: str,
    field: dict[str, Any],
    *,
    required: bool,
) -> tuple[str, tuple[type | Any, Any]]:
    """Generate a Pydantic field from a JSON schema field."""
    default_from_schema = field.get("default")
    return (
        field_name,
        (
            _map_pydantic_type(field_name, field),
            Field(
                default=... if required else default_from_schema,
                description=field.get("description", ""),
            ),
        ),
    )


def _map_pydantic_type(field_name: str, field: dict[str, Any]) -> type | Any:  # noqa: ANN401
    match field:
        case {"type": _}:
            return _map_single_pydantic_type(field_name, field)
        case {"oneOf": union_types} | {"anyOf": union_types}:
            types = [
                _map_single_pydantic_type(field_name, t, allow_nonetype=True) for t in union_types
            ]
            return Union[*types]
        case _:
            logger().debug(f"Unsupported JSON schema type: {field.get('type')}: {field}")
            return Any


def _map_single_pydantic_type(  # noqa: PLR0911
    field_name: str,
    field: dict[str, Any],
    *,
    allow_nonetype: bool = False,
) -> type | Any:  # noqa: ANN401
    match field.get("type"):
        case "string":
            return str
        case "integer":
            return int
        case "number":
            return float
        case "boolean":
            return bool
        case "array":
            item_type = _map_pydantic_type(field_name, field.get("items", {}))
            return list[item_type]
        case "object":
            return generate_pydantic_model_from_json_schema(f"{field_name}_model", field)
        case "null":
            if allow_nonetype:
                return None
            logger().debug(f"Null type is not allowed for a non-union field: {field_name}")
            return Any
        case _:
            logger().debug(f"Unsupported JSON schema type: {field.get('type')}: {field}")
            return Any
