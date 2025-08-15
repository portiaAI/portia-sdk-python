"""Configuration and client code for interactions with Model Context Protocol (MCP) servers.

This module provides a context manager for creating MCP ClientSessions, which are used to
interact with MCP servers. It supports SSE, stdio, and StreamableHTTP transports.

NB. The MCP Python SDK is asynchronous, so care must be taken when using MCP functionality
from this module in an async context.

Classes:
    SseMcpClientConfig: Configuration for an MCP client that connects via SSE.
    StdioMcpClientConfig: Configuration for an MCP client that connects via stdio.
    StreamableHttpMcpClientConfig: Configuration for an MCP client that connects via StreamableHTTP.
    McpClientConfig: The configuration to connect to an MCP server.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

import httpx
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class SseMcpClientConfig(BaseModel):
    """Configuration for an MCP client that connects via SSE."""

    server_name: str
    url: str
    headers: dict[str, Any] | None = None
    timeout: float = 5
    sse_read_timeout: float = 60 * 5
    tool_call_timeout_seconds: float | None = None


class StdioMcpClientConfig(BaseModel):
    """Configuration for an MCP client that connects via stdio."""

    server_name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    encoding: str = "utf-8"
    encoding_error_handler: Literal["strict", "ignore", "replace"] = "strict"
    tool_call_timeout_seconds: float | None = None

    @classmethod
    def from_raw(cls, config: str | dict[str, Any]) -> StdioMcpClientConfig:
        """Create a StdioMcpClientConfig from a string.

        This method is used to create a StdioMcpClientConfig from a string. It supports
        mcpServers and servers keys methods commonly used in MCP client configs.

        Args:
            config: The string or dict to parse.

        Returns:
            A StdioMcpClientConfig.

        Raises:
            ValueError: If the string is not valid JSON or does not contain a valid MCP config.

        """
        if isinstance(config, str):
            try:
                json_config = json.loads(config)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {config}") from e
        else:
            json_config = config

        if "mcpServers" in json_config:
            server_name = next(iter(json_config["mcpServers"].keys()))
            server_config = json_config["mcpServers"][server_name]
            return cls(
                server_name=server_name,
                command=server_config["command"],
                args=server_config["args"],
                env=server_config.get("env", None),
            )
        if "servers" in json_config:
            server_name = next(iter(json_config["servers"].keys()))
            server_config = json_config["servers"][server_name]
            return cls(
                server_name=server_name,
                command=server_config["command"],
                args=server_config["args"],
                env=server_config.get("env", None),
            )
        raise ValueError(f"Invalid MCP client config: {config}")


class StreamableHttpMcpClientConfig(BaseModel):
    """Configuration for an MCP client that connects via StreamableHTTP."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    server_name: str
    url: str
    headers: dict[str, Any] | None = None
    timeout: float = 30
    sse_read_timeout: float = 60 * 5
    terminate_on_close: bool = True
    auth: httpx.Auth | None = None
    tool_call_timeout_seconds: float | None = None


McpClientConfig = SseMcpClientConfig | StdioMcpClientConfig | StreamableHttpMcpClientConfig


@asynccontextmanager
async def get_mcp_session(mcp_client_config: McpClientConfig) -> AsyncIterator[ClientSession]:
    """Context manager for an MCP ClientSession.

    Args:
        mcp_client_config: The configuration to connect to an MCP server

    Returns:
        An MCP ClientSession

    """
    if isinstance(mcp_client_config, StdioMcpClientConfig):
        async with (
            stdio_client(
                StdioServerParameters(
                    command=mcp_client_config.command,
                    args=mcp_client_config.args,
                    env=mcp_client_config.env,
                    encoding=mcp_client_config.encoding,
                    encoding_error_handler=mcp_client_config.encoding_error_handler,
                ),
            ) as stdio_transport,
            ClientSession(*stdio_transport) as session,
        ):
            await session.initialize()
            yield session
    elif isinstance(mcp_client_config, SseMcpClientConfig):
        async with (
            sse_client(
                url=mcp_client_config.url,
                headers=mcp_client_config.headers,
                timeout=mcp_client_config.timeout,
                sse_read_timeout=mcp_client_config.sse_read_timeout,
            ) as sse_transport,
            ClientSession(*sse_transport) as session,
        ):
            await session.initialize()
            yield session
    elif isinstance(mcp_client_config, StreamableHttpMcpClientConfig):
        async with (
            streamablehttp_client(
                url=mcp_client_config.url,
                headers=mcp_client_config.headers,
                timeout=timedelta(seconds=mcp_client_config.timeout),
                sse_read_timeout=timedelta(seconds=mcp_client_config.sse_read_timeout),
                terminate_on_close=mcp_client_config.terminate_on_close,
                auth=mcp_client_config.auth,
            ) as streamablehttp_transport,
            ClientSession(streamablehttp_transport[0], streamablehttp_transport[1]) as session,
        ):
            await session.initialize()
            yield session
