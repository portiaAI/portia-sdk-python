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
import threading
from enum import StrEnum
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import httpx
from jsonref import replace_refs
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    model_serializer,
    model_validator,
)
from pydantic_core import PydanticUndefined

from portia.cloud import PortiaCloudClient
from portia.errors import DuplicateToolError, InvalidToolDescriptionError, ToolNotFoundError
from portia.logger import logger
from portia.mcp_session import (
    McpClientConfig,
    SseMcpClientConfig,
    StdioMcpClientConfig,
    StreamableHttpMcpClientConfig,
    get_mcp_session,
)
from portia.open_source_tools.calculator_tool import CalculatorTool
from portia.open_source_tools.crawl_tool import CrawlTool
from portia.open_source_tools.extract_tool import ExtractTool
from portia.open_source_tools.image_understanding_tool import ImageUnderstandingTool
from portia.open_source_tools.llm_tool import LLMTool
from portia.open_source_tools.local_file_reader_tool import FileReaderTool
from portia.open_source_tools.local_file_writer_tool import FileWriterTool
from portia.open_source_tools.map_tool import MapTool
from portia.open_source_tools.search_tool import SearchTool
from portia.open_source_tools.weather import WeatherTool
from portia.tool import PortiaMcpTool, PortiaRemoteTool, Tool

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Iterator, Sequence

    import mcp
    from pydantic_core.core_schema import SerializerFunctionWrapHandler

    from portia.config import Config


class ToolRegistry:
    """ToolRegistry is the base class for managing tools.

    This class implements the essential methods for interacting with tool registries, including
    registering, retrieving, and listing tools. Specific tool registries can override these methods
    and provide additional functionality.

    Methods:
        with_tool(tool: Tool, *, overwrite: bool = False) -> None:
            Inserts a new tool.
        replace_tool(tool: Tool) -> None:
            Replaces a tool with a new tool in the current registry.
            NB. This is a shortcut for `with_tool(tool, overwrite=True)`.
        get_tool(tool_id: str) -> Tool:
            Retrieves a tool by its ID.
        get_tools() -> list[Tool]:
            Retrieves all tools in the registry.
        match_tools(query: str | None = None, tool_ids: list[str] | None = None) -> list[Tool]:
            Optionally, retrieve tools that match a given query and tool_ids.
        filter_tools(predicate: Callable[[Tool], bool]) -> ToolRegistry:
            Create a new tool registry with only the tools that match the predicate. Useful to
            implement tool exclusions.
        with_tool_description(
            tool_id: str,
            updated_description: str,
            *,
            overwrite: bool = False,
        ) -> ToolRegistry:
            Extend or override the description of a tool in the registry.

    """

    def __init__(self, tools: dict[str, Tool] | Sequence[Tool] | None = None) -> None:
        """Initialize the tool registry with a sequence or dictionary of tools.

        Args:
            tools (dict[str, Tool] | Sequence[Tool]): A sequence of tools or a
              dictionary of tool IDs to tools.

        """
        if tools is None:
            self._tools = {}
        elif not isinstance(tools, dict):
            self._tools = {tool.id: tool for tool in tools}
        else:
            self._tools = tools

    def with_tool(self, tool: Tool, *, overwrite: bool = False) -> None:
        """Update a tool based on tool ID or inserts a new tool.

        Args:
            tool (Tool): The tool to be added or updated.
            overwrite (bool): Whether to overwrite an existing tool with the same ID.

        Returns:
            None: The tool registry is updated in place.

        """
        if tool.id in self._tools and not overwrite:
            raise DuplicateToolError(tool.id)
        self._tools[tool.id] = tool

    def replace_tool(self, tool: Tool) -> None:
        """Replace a tool with a new tool.

        Args:
            tool (Tool): The tool to replace the existing tool with.

        Returns:
            None: The tool registry is updated in place.

        """
        self.with_tool(tool, overwrite=True)

    def get_tool(self, tool_id: str) -> Tool:
        """Retrieve a tool's information.

        Args:
            tool_id (str): The ID of the tool to retrieve.

        Returns:
            Tool: The requested tool.

        Raises:
            ToolNotFoundError: If the tool with the given ID does not exist.

        """
        if tool_id not in self._tools:
            raise ToolNotFoundError(tool_id)
        return self._tools[tool_id]

    def get_tools(self) -> list[Tool]:
        """Get all tools registered with the registry.

        Returns:
            list[Tool]: A list of all tools in the registry.

        """
        return list(self._tools.values())

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

    def filter_tools(self, predicate: Callable[[Tool], bool]) -> ToolRegistry:
        """Filter the tools in the registry based on a predicate.

        Args:
            predicate (Callable[[Tool], bool]): A predicate to filter the tools.

        Returns:
            Self: A new ToolRegistry with the filtered tools.

        """
        return ToolRegistry({tool.id: tool for tool in self._tools.values() if predicate(tool)})

    def with_tool_description(
        self, tool_id: str, updated_description: str, *, overwrite: bool = False
    ) -> ToolRegistry:
        """Update a tool with an extension or override of the tool description.

        Args:
            tool_id (str): The id of the tool to update.
            updated_description (str): The tool description to update. If `overwrite` is False, this
                will extend the existing tool description, otherwise, the entire tool description
                will be updated.
            overwrite (bool): Whether to update or extend the existing tool description.

        Returns:
            Self: The tool registry is updated in place and returned.

        Particularly useful for customising tools in MCP servers for usecases. A deep copy is made
        of the underlying tool such that the tool description is only updated within this registry.
        Logs a warning if the tool is not found.

        """
        try:
            tool = self.get_tool(tool_id=tool_id)
            new_description = (
                updated_description if overwrite else f"{tool.description}. {updated_description}"
            )
            self.replace_tool(tool.model_copy(update={"description": new_description}, deep=False))
        except ToolNotFoundError:
            logger().warning(f"Unknown tool ID: {tool_id}. Description was not edited.")
        return self

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
            ToolRegistry: A new tool registry containing tools from both registries.

        """
        return self._add(other)

    def __iter__(self) -> Iterator[Tool]:
        """Iterate over the tools in the registry."""
        return iter(self._tools.values())

    def __len__(self) -> int:
        """Return the number of tools in the registry."""
        return len(self._tools)

    def __contains__(self, tool_id: str) -> bool:
        """Check if a tool is in the registry."""
        return tool_id in self._tools

    def _add(self, other: ToolRegistry | list[Tool]) -> ToolRegistry:
        """Add a tool registry or Tool list to the current registry."""
        other_registry = other if isinstance(other, ToolRegistry) else ToolRegistry(other)
        self_tools = self.get_tools()
        other_tools = other_registry.get_tools()
        tools = {}
        for tool in [*self_tools, *other_tools]:
            if tool.id in tools:
                logger().warning(
                    f"Duplicate tool ID found: {tool.id!s}. Unintended behavior may occur.",
                )
            tools[tool.id] = tool

        return ToolRegistry(tools)


class InMemoryToolRegistry(ToolRegistry):
    """Provides a simple in-memory tool registry.

    This class stores tools in memory, allowing for quick access without persistence.

    Warning: This registry is DEPRECATED. Use ToolRegistry instead.
    """

    @classmethod
    def from_local_tools(cls, tools: Sequence[Tool]) -> InMemoryToolRegistry:
        """Easily create a local tool registry from a sequence of tools.

        Args:
            tools (Sequence[Tool]): A sequence of tools to initialize the registry.

        Returns:
            InMemoryToolRegistry: A new in-memory tool registry.

        """
        return cls(tools)


class PortiaToolRegistry(ToolRegistry):
    """Provides access to Portia tools.

    This class interacts with the Portia API to retrieve and manage tools.
    """

    EXCLUDED_BY_DEFAULT_TOOL_REGEXS: frozenset[str] = frozenset()

    def __init__(
        self,
        config: Config | None = None,
        client: httpx.Client | None = None,
        tools: dict[str, Tool] | Sequence[Tool] | None = None,
    ) -> None:
        """Initialize the PortiaToolRegistry with the given configuration.

        Args:
            config (Config | None): The configuration containing the API key and endpoint.
            client (httpx.Client | None): An optional httpx client to use. If not provided, a new
              client will be created.
            tools (dict[str, Tool] | None): A dictionary of tool IDs to tools to create the
              registry with. If not provided, all tools will be loaded from the Portia API.

        """
        if tools is not None:
            super().__init__(tools)
        elif client is not None:
            super().__init__(self._load_tools(client))
        elif config is not None:
            client = PortiaCloudClient.new_client(config)
            super().__init__(self._load_tools(client))
        else:
            raise ValueError("Either config, client or tools must be provided")

    def with_default_tool_filter(self) -> PortiaToolRegistry:
        """Create a PortiaToolRegistry with a default tool filter."""

        def default_tool_filter(tool: Tool) -> bool:
            """Filter out tools that match the default tool regexes."""
            return not any(
                re.match(regex, tool.id) for regex in self.EXCLUDED_BY_DEFAULT_TOOL_REGEXS
            )

        return PortiaToolRegistry(tools=self.filter_tools(default_tool_filter).get_tools())

    @classmethod
    def _load_tools(cls, client: httpx.Client) -> dict[str, Tool]:
        """Load the tools from the API into the into the internal storage."""
        response = client.get(
            url="/api/v0/tools/descriptions-v2/",
        )
        if response.status_code == httpx.codes.NOT_FOUND:
            response = client.get(
                url="/api/v0/tools/descriptions/",
            )
            response_tools = response.json()
        else:
            response.raise_for_status()
            response_tools = response.json().get("tools", [])
            for error in response.json().get("errors", []):
                logger().warning(
                    f"Error loading Portia Cloud tool for app: {error['app_name']}: "
                    f"{error['error']}"
                )
        tools = {}
        for raw_tool in response_tools:
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
                client=client,
            )
            tools[raw_tool["tool_id"]] = tool
        return tools


class McpToolRegistry(ToolRegistry):
    """Provides access to tools within a Model Context Protocol (MCP) server.

    See https://modelcontextprotocol.io/introduction for more information on MCP.
    """

    @classmethod
    def from_sse_connection(
        cls,
        server_name: str,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
        tool_list_read_timeout: float | None = None,
        tool_call_timeout_seconds: float | None = None,
    ) -> McpToolRegistry:
        """Create a new MCPToolRegistry using an SSE connection (Sync version)."""
        config = SseMcpClientConfig(
            server_name=server_name,
            url=url,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
            tool_call_timeout_seconds=tool_call_timeout_seconds,
        )
        tools = cls._load_tools(config, read_timeout=tool_list_read_timeout)
        return cls(tools)

    @classmethod
    async def from_sse_connection_async(
        cls,
        server_name: str,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: float = 5,  # noqa: ASYNC109
        sse_read_timeout: float = 60 * 5,
        tool_list_read_timeout: float | None = None,
        tool_call_timeout_seconds: float | None = None,
    ) -> McpToolRegistry:
        """Create a new MCPToolRegistry using an SSE connection (Async version)."""
        config = SseMcpClientConfig(
            server_name=server_name,
            url=url,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
            tool_call_timeout_seconds=tool_call_timeout_seconds,
        )
        tools = await cls._load_tools_async(config, read_timeout=tool_list_read_timeout)
        return cls(tools)

    @classmethod
    def from_stdio_connection(
        cls,
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        encoding: str = "utf-8",
        encoding_error_handler: Literal["strict", "ignore", "replace"] = "strict",
        tool_list_read_timeout: float | None = None,
        tool_call_timeout_seconds: float | None = None,
    ) -> McpToolRegistry:
        """Create a new MCPToolRegistry using a stdio connection (Sync version)."""
        config = StdioMcpClientConfig(
            server_name=server_name,
            command=command,
            args=args if args is not None else [],
            env=env,
            encoding=encoding,
            encoding_error_handler=encoding_error_handler,
            tool_call_timeout_seconds=tool_call_timeout_seconds,
        )
        tools = cls._load_tools(config, read_timeout=tool_list_read_timeout)
        return cls(tools)

    @classmethod
    def from_stdio_connection_raw(
        cls,
        config: str | dict[str, Any],
        tool_list_read_timeout: float | None = None,
        tool_call_timeout_seconds: float | None = None,
    ) -> McpToolRegistry:
        """Create a new MCPToolRegistry using a stdio connection from a string.

        Parses commonly used mcp client config formats.

        Args:
            config: The string or dict to parse.
            tool_list_read_timeout: The timeout for the request.
            tool_call_timeout_seconds: The timeout for the tool call.

        Returns:
            A McpToolRegistry.

        """
        parsed_config = StdioMcpClientConfig.from_raw(config)
        parsed_config.tool_call_timeout_seconds = tool_call_timeout_seconds
        tools = cls._load_tools(parsed_config, read_timeout=tool_list_read_timeout)
        return cls(tools)

    @classmethod
    async def from_stdio_connection_async(
        cls,
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        encoding: str = "utf-8",
        encoding_error_handler: Literal["strict", "ignore", "replace"] = "strict",
        tool_list_read_timeout: float | None = None,
        tool_call_timeout_seconds: float | None = None,
    ) -> McpToolRegistry:
        """Create a new MCPToolRegistry using a stdio connection (Async version)."""
        config = StdioMcpClientConfig(
            server_name=server_name,
            command=command,
            args=args if args is not None else [],
            env=env,
            encoding=encoding,
            encoding_error_handler=encoding_error_handler,
            tool_call_timeout_seconds=tool_call_timeout_seconds,
        )
        tools = await cls._load_tools_async(config, read_timeout=tool_list_read_timeout)
        return cls(tools)

    @classmethod
    def from_streamable_http_connection(
        cls,
        server_name: str,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: float = 30,
        sse_read_timeout: float = 60 * 5,
        *,
        terminate_on_close: bool = True,
        auth: httpx.Auth | None = None,
        tool_list_read_timeout: float | None = None,
        tool_call_timeout_seconds: float | None = None,
    ) -> McpToolRegistry:
        """Create a new MCPToolRegistry using a StreamableHTTP connection (Sync version)."""
        config = StreamableHttpMcpClientConfig(
            server_name=server_name,
            url=url,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
            terminate_on_close=terminate_on_close,
            auth=auth,
            tool_call_timeout_seconds=tool_call_timeout_seconds,
        )
        tools = cls._load_tools(config, read_timeout=tool_list_read_timeout)
        return cls(tools)

    @classmethod
    async def from_streamable_http_connection_async(
        cls,
        server_name: str,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: float = 30,  # noqa: ASYNC109
        sse_read_timeout: float = 60 * 5,
        *,
        terminate_on_close: bool = True,
        auth: httpx.Auth | None = None,
        tool_list_read_timeout: float | None = None,
        tool_call_timeout_seconds: float | None = None,
    ) -> McpToolRegistry:
        """Create a new MCPToolRegistry using a StreamableHTTP connection (Async version)."""
        config = StreamableHttpMcpClientConfig(
            server_name=server_name,
            url=url,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
            terminate_on_close=terminate_on_close,
            auth=auth,
            tool_call_timeout_seconds=tool_call_timeout_seconds,
        )
        tools = await cls._load_tools_async(config, read_timeout=tool_list_read_timeout)
        return cls(tools)

    @classmethod
    def _load_tools(
        cls,
        mcp_client_config: McpClientConfig,
        *,
        read_timeout: float | None = None,
    ) -> list[PortiaMcpTool]:
        """Sync version to load tools from an MCP server."""
        T = TypeVar("T")

        def _run_async_in_new_loop(coro: Coroutine[Any, Any, T]) -> T:
            """Run an asynchronous coroutine in a new event loop within a separate thread.

            Args:
                coro: The coroutine to execute.
                read_timeout (float): The timeout for the request.

            Returns:
                The result returned by the coroutine.

            """
            result_container: dict[str, T | Exception] = {}

            def runner() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result_container["result"] = loop.run_until_complete(coro)
                except Exception as e:  # noqa: BLE001
                    result_container["error"] = e
                finally:
                    loop.close()

            thread = threading.Thread(target=runner)
            thread.start()
            thread.join()
            if isinstance(result_container.get("error"), Exception):
                raise result_container["error"]  # type: ignore  # noqa: PGH003
            return result_container["result"]  # type: ignore  # noqa: PGH003

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:  # pragma: no cover
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return _run_async_in_new_loop(
            cls._load_tools_async(mcp_client_config, read_timeout=read_timeout),
        )

    @classmethod
    async def _load_tools_async(
        cls,
        mcp_client_config: McpClientConfig,
        *,
        read_timeout: float | None = None,
    ) -> list[PortiaMcpTool]:
        """Async version to load tools from an MCP server.

        The MCP client session doesn't support timeouts for list_tools, so we use
        our own implementation to wrap the request in a timeout.

        Args:
            mcp_client_config (McpClientConfig): The MCP client configuration.
            read_timeout (float): The timeout for the request.

        Returns:
            list[PortiaMcpTool]: The list of Portia MCP tools.

        """

        async def _inner() -> list[PortiaMcpTool]:
            """Inner function to wrap in wait_for to implement timeout."""
            async with get_mcp_session(mcp_client_config) as session:
                logger().debug("Fetching tools from MCP server")
                tools = await session.list_tools()
                logger().debug(f"Got {len(tools.tools)} tools from MCP server")
                return [
                    portia_tool
                    for tool in tools.tools
                    if (
                        portia_tool := cls._portia_tool_from_mcp_tool(
                            tool,
                            mcp_client_config,
                        )
                    )
                    is not None
                ]

        return await asyncio.wait_for(_inner(), timeout=read_timeout)

    @classmethod
    def _portia_tool_from_mcp_tool(
        cls,
        mcp_tool: mcp.Tool,
        mcp_client_config: McpClientConfig,
    ) -> PortiaMcpTool | None:
        """Conversion of a remote MCP server tool to a Portia tool."""
        tool_name_snake_case = re.sub(r"[^a-zA-Z0-9]+", "_", mcp_tool.name)

        description = (
            mcp_tool.description
            if mcp_tool.description is not None
            else f"{mcp_tool.name} tool from {mcp_client_config.server_name}"
        )
        try:
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
        except (ValidationError, InvalidToolDescriptionError) as e:
            logger().warning(
                f"Error creating Portia Tool object for tool from {mcp_client_config.server_name} "
                f"with name {mcp_tool.name}: {e}"
            )
            return None


class DefaultToolRegistry(ToolRegistry):
    """A registry providing a default set of tools.

    This includes the following tools:
    - All open source tools that don't require API keys
    - Search, map, extract, and crawl tools if you have a Tavily API key
    - Weather tool if you have an OpenWeatherMap API key
    - Portia cloud tools if you have a Portia cloud API key
    """

    def __init__(self, config: Config) -> None:
        """Initialize the default tool registry with the given configuration."""
        tools = [
            CalculatorTool(),
            LLMTool(),
            FileWriterTool(),
            FileReaderTool(),
            ImageUnderstandingTool(),
        ]
        if os.getenv("TAVILY_API_KEY"):
            tools.append(SearchTool())
            tools.append(MapTool())
            tools.append(ExtractTool())
            tools.append(CrawlTool())
        if os.getenv("OPENWEATHERMAP_API_KEY"):
            tools.append(WeatherTool())

        if config.portia_api_key:
            tools.extend(PortiaToolRegistry(config).with_default_tool_filter().get_tools())

        super().__init__(tools)


class GeneratedBaseModel(BaseModel):
    """BaseModel that is generated from a JSON schema.

    Handles serialization of fields that must omit None values: fields that are not required in
    the JSON schema, but that are not nullable. Pydantic has no concept of an omissible field,
    so we must for it to be nullable and then make sure we don't serialize None values.
    """

    _fields_must_omit_none_on_serialize: ClassVar[list[str]] = []

    def __init_subclass__(cls) -> None:
        """Ensure omissible fields are isolated between models."""
        super().__init_subclass__()
        cls._fields_must_omit_none_on_serialize = []

    @model_serializer(mode="wrap")
    def serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Serialize the model to a dictionary, excluding fields for which we must omit None."""
        ser = handler(self)
        for field in self._fields_must_omit_none_on_serialize:
            if field in ser and (ser[field] is PydanticUndefined or ser[field] is None):
                del ser[field]
        return ser

    @classmethod
    def extend_exclude_unset_fields(cls, fields: list[str]) -> None:
        """Extend the list of fields to exclude from serialization."""
        cls._fields_must_omit_none_on_serialize.extend(fields)


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


def _additional_properties_validator(
    self: BaseModelT,
    extras_schema: type[BaseModel],
    field_name: str,
) -> BaseModelT:
    """Validate that extra properties against a schema."""
    if self.model_extra is None:
        return self  # pragma: no cover
    for key, value in self.model_extra.items():  # all unknowns live here
        try:
            extras_schema.model_validate({field_name: value})
        except ValidationError as e:
            raise ValueError(f"Extra field {key!r} must match the schema: {e}") from e
    return self


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
    schema_without_refs = replace_refs(json_schema, proxies=False)

    # Extract properties and required fields
    properties = schema_without_refs.get("properties", {})  # type: ignore  # noqa: PGH003
    required = set(schema_without_refs.get("required", []))  # type: ignore  # noqa: PGH003

    non_nullable_omissible_fields = [
        field_name
        for field_name, field in properties.items()
        if (
            "default" not in field
            and field_name not in required
            and not _is_nullable_field(field_name, field)
        )
    ]
    additional_properties: bool | dict[str, Any] = schema_without_refs.get(  # type: ignore  # noqa: PGH003
        "additionalProperties", False
    )
    extra_allowed = additional_properties is True or isinstance(additional_properties, dict)
    config_dict: ConfigDict | None = None
    pydantic_validator: Any = None
    if extra_allowed:
        config_dict = ConfigDict(extra="allow")
    if isinstance(additional_properties, dict):
        # additionalProperties as a dict is a JSON schema which any additional properties must match
        extras_schema = generate_pydantic_model_from_json_schema(
            f"{model_name}_extras",
            {"type": "object", "properties": {"extra_property": additional_properties}},
        )
        validator = partial(
            _additional_properties_validator,
            extras_schema=extras_schema,
            field_name="extra_property",
        )
        pydantic_validator = model_validator(mode="after")(validator)

    # Define fields for the model
    fields = dict(
        [
            _generate_field(
                key,
                value,
                required=key in required,
                force_nullable=key in non_nullable_omissible_fields,
            )
            for key, value in properties.items()
        ],
    )

    # Create the Pydantic model dynamically
    model = create_model(
        model_name,
        __base__=GeneratedBaseModel,
        __config__=config_dict,
        __validators__={"extra": pydantic_validator} if pydantic_validator else None,
        **fields,  # type: ignore  # noqa: PGH003
    )
    model.extend_exclude_unset_fields(non_nullable_omissible_fields)
    return model


def _is_nullable_field(field_name: str, field: dict[str, Any]) -> bool:
    """Check if a field is nullable."""
    python_type = _map_pydantic_type(field_name, field)
    if get_origin(python_type) is Union:
        return type(None) in get_args(python_type)
    return False


def _generate_field(
    field_name: str,
    field: dict[str, Any],
    *,
    required: bool,
    force_nullable: bool,
) -> tuple[str, tuple[type | Any, Any]]:
    """Generate a Pydantic field from a JSON schema field."""
    default_from_schema = field.get("default")
    field_type = _map_pydantic_type(field_name, field)
    if force_nullable:
        field_type = field_type | None

    field_kwargs: dict[str, Any] = {
        "default": ... if required else default_from_schema,
        "description": field.get("description", ""),
    }
    if "minLength" in field:
        field_kwargs["min_length"] = field["minLength"]
    if "maxLength" in field:
        field_kwargs["max_length"] = field["maxLength"]

    return (
        field_name,
        (
            field_type,
            Field(**field_kwargs),
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
            return Union[*types]  # pyright: ignore[reportInvalidTypeForm, reportInvalidTypeArguments]
        case _:
            logger().debug(f"Unsupported JSON schema type: {field.get('type')}: {field}")
            return Any


def _map_single_pydantic_type(  # noqa: C901, PLR0911
    field_name: str,
    field: dict[str, Any],
    *,
    allow_nonetype: bool = False,
) -> type | Any:  # noqa: ANN401
    match field.get("type"):
        case "string":
            if field.get("enum"):
                return StrEnum(
                    field_name,
                    {v.upper() if v else "__EMPTY_STRING": v for v in field.get("enum", [])},
                )
            return str
        case "integer":
            return int
        case "number":
            return float | int
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
        case [*types]:
            types = [
                _map_single_pydantic_type(
                    field_name,
                    {"type": t} if isinstance(t, str) else t,
                    allow_nonetype=True,
                )
                for t in types
            ]
            return Union[*types]  # pyright: ignore[reportInvalidTypeForm, reportInvalidTypeArguments]
        case _:
            logger().debug(f"Unsupported JSON schema type: {field.get('type')}: {field}")
            return Any
