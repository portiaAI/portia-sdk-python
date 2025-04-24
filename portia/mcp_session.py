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

import asyncio
import json
import threading
import webbrowser
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import timedelta
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal, Optional  # noqa: UP035
from urllib.parse import parse_qs, urlparse

import httpx
import keyring
import requests
from authlib.integrations.requests_client import OAuth2Session
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
    use_oauth: bool = False

    @property
    def keyring_app(self) -> str:
        return f"mcp-{self.server_name}"


class StdioMcpClientConfig(BaseModel):
    """Configuration for an MCP client that connects via stdio."""

    server_name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    encoding: str = "utf-8"
    encoding_error_handler: Literal["strict", "ignore", "replace"] = "strict"


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
        async with get_http_session(mcp_client_config) as session:
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


class OAuthServerMetadata(BaseModel):
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str
    response_types_supported: list[str]
    response_modes_supported: list[str]
    grant_types_supported: list[str]
    token_endpoint_auth_methods_supported: list[str]
    revocation_endpoint: str
    code_challenge_methods_supported: list[str]


class OAuthClientRegistration(BaseModel):
    client_id: str
    redirect_uris: list[str]
    client_name: str
    client_uri: str
    grant_types: list[str]
    response_types: list[str]
    token_endpoint_auth_method: str
    registration_client_uri: str
    client_id_issued_at: int


LOCAL_PORT: int = 17249

LOCAL_REDIRECT_URI: str = f"http://127.0.0.1:{LOCAL_PORT}/callback"
KEYRING_USER: str = "me@example.com"
# ————————————


def load_token(mcp_client_config: SseMcpClientConfig) -> dict[str, Any]:
    raw: str | None = keyring.get_password(mcp_client_config.keyring_app, KEYRING_USER)
    if not raw:
        raise KeyError("No token found")
    return json.loads(raw)


def save_token(mcp_client_config: SseMcpClientConfig, token: dict[str, Any]) -> None:
    print(f"Saving token: {token}")  # noqa: T201
    keyring.set_password(mcp_client_config.keyring_app, KEYRING_USER, json.dumps(token))


def authorize_flow(mcp_client_config: SseMcpClientConfig) -> None:
    # 1) Create an OAuth2 session (with PKCE built-in if you set code_challenge_method)
    server_metadata: OAuthServerMetadata = fetch_oauth_server_metadata(mcp_client_config)
    client_registration: OAuthClientRegistration = register_client(server_metadata)
    client: OAuth2Session = OAuth2Session(
        client_id=client_registration.client_id,
        redirect_uri=LOCAL_REDIRECT_URI,
        code_challenge_method="S256",  # comment out if not using PKCE
    )

    # 2) Get the authorization URL
    auth_url: str
    state: str
    auth_url, state = client.create_authorization_url(server_metadata.authorization_endpoint)
    print("Go to this URL in your browser to authorize:")
    print(auth_url)
    webbrowser.open(auth_url)

    # 3) Spin up a simple HTTP server to catch the callback
    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            # Parse the incoming URL for code
            url = urlparse(self.path)
            if url.path != "/callback":
                self.send_error(404)
                return

            qs: dict[str, list[str]] = parse_qs(url.query)
            code: str | None = qs.get("code", [None])[0]
            if not code:
                self.send_error(400, "Missing code")
                return

            # 4) Exchange code for token
            token: dict[str, Any] = client.fetch_token(
                server_metadata.token_endpoint,
                code=code,
            )
            token["oauth_client"] = client_registration.model_dump(mode="json")
            save_token(mcp_client_config, token)

            # 5) Notify the user in the browser
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authentication successful!</h1>You can close this window.")

            # 6) Shut down the server cleanly
            threading.Thread(target=self.server.shutdown).start()

    httpd: HTTPServer = HTTPServer(("127.0.0.1", LOCAL_PORT), CallbackHandler)
    print(f"Waiting for OAuth callback on http://127.0.0.1:{LOCAL_PORT}/callback …")
    httpd.serve_forever()


def fetch_oauth_server_metadata(mcp_client_config: SseMcpClientConfig) -> OAuthServerMetadata:
    """
    Fetches the OAuth authorization server metadata from the MCP GitHub OAuth server.
    Returns the JSON response containing server configuration.
    """

    response = requests.get(
        f"{mcp_client_config.url}/.well-known/oauth-authorization-server",
    )
    response.raise_for_status()
    return OAuthServerMetadata.model_validate_json(response.content)


def register_client(metadata: OAuthServerMetadata) -> OAuthClientRegistration:
    response = requests.post(
        metadata.registration_endpoint,
        json={
            "redirect_uris": [LOCAL_REDIRECT_URI],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "client_name": "Sam MCP local",
            "client_uri": "https://github.com/sam-portia",
        },
    )
    response.raise_for_status()
    return OAuthClientRegistration.model_validate_json(response.content)


def get_token(mcp_client_config: SseMcpClientConfig) -> dict[str, Any]:
    try:
        return load_token(mcp_client_config)
    except KeyError:
        authorize_flow(mcp_client_config)
        return load_token(mcp_client_config)


@asynccontextmanager
async def _get_session(
    mcp_client_config: SseMcpClientConfig, token: dict[str, Any] | None
) -> AsyncIterator[ClientSession]:
    if token:
        headers = {
            "Authorization": f"Bearer {token['access_token']}",
            **(mcp_client_config.headers or {}),
        }
    else:
        headers = mcp_client_config.headers
    async with sse_client(  # noqa: SIM117
        url=f"{mcp_client_config.url}/sse",
        timeout=mcp_client_config.timeout,
        sse_read_timeout=mcp_client_config.sse_read_timeout,
        headers=headers,
    ) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            yield session
    


@asynccontextmanager
async def get_http_session(mcp_client_config: SseMcpClientConfig) -> AsyncIterator[ClientSession]:
    token = get_token(mcp_client_config) if mcp_client_config.use_oauth else None
    try:
        async with _get_session(mcp_client_config, token) as session:
            yield session
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401 and mcp_client_config.use_oauth:
            # Use the refresh token to get a new access token
            server_metadata = fetch_oauth_server_metadata(mcp_client_config)
            client = OAuth2Session(
                client_id=token["oauth_client"]["client_id"],
                token=token,
                token_endpoint=server_metadata.token_endpoint,
            )
            new_token = client.refresh_token(server_metadata.token_endpoint)
            save_token(mcp_client_config, new_token)
            # Try again with the new token
            async with _get_session(mcp_client_config, new_token) as session:
                yield session
        else:
            raise


async def main() -> None:
    conf = SseMcpClientConfig(
        server_name="my_oauth_mcp",
        url="https://mcp-github-oauth.sam-f86.workers.dev",
        use_oauth=True,
    )
    async with get_mcp_session(conf) as session:
        tools = await session.list_tools()
        print(tools.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
