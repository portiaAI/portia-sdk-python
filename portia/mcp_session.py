"""Configuration and client code for interactions with Model Context Protocol (MCP) servers.

This module provides a context manager for creating MCP ClientSessions, which are used to
interact with MCP servers. It supports both the SSE and stdio transports.

NB. The MCP Python SDK is asynchronous, so care must be taken when using MCP functionality
from this module in an async context.

Classes:
    SseMcpClientConfig: Configuration for an MCP client that connects via SSE.
    StdioMcpClientConfig: Configuration for an MCP client that connects via stdio.
    McpClientConfig: The configuration to connect to an MCP server.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class SseMcpClientConfig(BaseModel):
    """Configuration for an MCP client that connects via SSE."""

    server_name: str
    url: str
    headers: dict[str, Any] | None = None
    timeout: float = 5
    sse_read_timeout: float = 60 * 5


class StdioMcpClientConfig(BaseModel):
    """Configuration for an MCP client that connects via stdio."""

    server_name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    encoding: str = "utf-8"
    encoding_error_handler: Literal["strict", "ignore", "replace"] = "strict"


McpClientConfig = SseMcpClientConfig | StdioMcpClientConfig


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
