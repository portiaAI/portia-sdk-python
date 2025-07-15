"""Tool to crawl websites."""

from __future__ import annotations

import os
from typing import Any, NoReturn

import httpx
from pydantic import BaseModel, Field

from portia.errors import ToolHardError, ToolSoftError
from portia.tool import Tool, ToolRunContext

# Constants for default values
DEFAULT_MAX_DEPTH = 1
DEFAULT_MAX_BREADTH = 20
DEFAULT_LIMIT = 50


class CrawlToolSchema(BaseModel):
    """Input for CrawlTool."""

    url: str = Field(
        ..., description="The root URL to begin the crawl (e.g., 'https://docs.tavily.com')"
    )
    instructions: str | None = Field(
        default=None,
        description=(
            "Natural language instructions for the crawler "
            "(e.g., 'Find all pages on the Python SDK')"
        ),
    )
    max_depth: int = Field(
        default=DEFAULT_MAX_DEPTH,
        description=(
            "Max depth of the crawl. Defines how far from the base URL the crawler can explore"
        ),
        ge=1,
        le=5,
    )
    max_breadth: int = Field(
        default=DEFAULT_MAX_BREADTH,
        description="Max number of links to follow per level of the tree (i.e., per page)",
        ge=1,
        le=100,
    )
    limit: int = Field(
        default=DEFAULT_LIMIT,
        description="Total number of links the crawler will process before stopping",
        ge=1,
        le=500,
    )
    select_paths: list[str] | None = Field(
        default=None,
        description=(
            "Regex patterns to select only URLs with specific path patterns "
            "(e.g., ['/docs/.*', '/api/v1.*'])"
        ),
    )
    select_domains: list[str] | None = Field(
        default=None,
        description=(
            "Regex patterns to select crawling to specific domains or subdomains "
            "(e.g., ['^docs\\.example\\.com$'])"
        ),
    )
    exclude_paths: list[str] | None = Field(
        default=None,
        description=(
            "Regex patterns to exclude URLs with specific path patterns "
            "(e.g., ['/private/.*', '/admin/.*'])"
        ),
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description=(
            "Regex patterns to exclude specific domains or subdomains from crawling "
            "(e.g., ['^private\\.example\\.com$'])"
        ),
    )
    allow_external: bool = Field(
        default=False, description="Whether to allow following links that go to external domains"
    )


class CrawlTool(Tool[str]):
    """Crawls websites using graph-based traversal tool."""

    id: str = "crawl_tool"
    name: str = "Crawl Tool"
    description: str = (
        "Crawls websites using graph-based website traversal tool that can explore "
        "hundreds of paths in parallel with built-in extraction and intelligent discovery. "
        "Provide a starting URL and optional instructions for what to find, and the tool will "
        "navigate and extract relevant content from multiple pages. Supports depth control, "
        "domain filtering, and path selection for comprehensive site exploration."
    )
    args_schema: type[BaseModel] = CrawlToolSchema
    output_schema: tuple[str, str] = ("str", "str: crawled content and discovered pages")

    def run(
        self,
        _: ToolRunContext,
        url: str,
        instructions: str | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_breadth: int = DEFAULT_MAX_BREADTH,
        limit: int = DEFAULT_LIMIT,
        select_paths: list[str] | None = None,
        select_domains: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        allow_external: bool = False,
    ) -> str:
        """Run the crawl tool."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key or api_key == "":
            raise ToolHardError("TAVILY_API_KEY is required to use crawl")

        payload = self._build_payload(
            url=url,
            instructions=instructions,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
            select_paths=select_paths,
            select_domains=select_domains,
            exclude_paths=exclude_paths,
            exclude_domains=exclude_domains,
            allow_external=allow_external,
        )

        return self._make_api_request(api_key, payload)

    def _build_payload(
        self,
        url: str,
        instructions: str | None,
        max_depth: int,
        max_breadth: int,
        limit: int,
        select_paths: list[str] | None,
        select_domains: list[str] | None,
        exclude_paths: list[str] | None,
        exclude_domains: list[str] | None,
        allow_external: bool,
    ) -> dict[str, Any]:
        """Build the API payload with optional parameters."""
        payload: dict[str, Any] = {"url": url}

        # Add optional parameters only when provided
        if instructions is not None:
            payload["instructions"] = instructions
        if max_depth != DEFAULT_MAX_DEPTH:
            payload["max_depth"] = max_depth
        if max_breadth != DEFAULT_MAX_BREADTH:
            payload["max_breadth"] = max_breadth
        if limit != DEFAULT_LIMIT:
            payload["limit"] = limit
        if select_paths is not None:
            payload["select_paths"] = select_paths
        if select_domains is not None:
            payload["select_domains"] = select_domains
        if exclude_paths is not None:
            payload["exclude_paths"] = exclude_paths
        if exclude_domains is not None:
            payload["exclude_domains"] = exclude_domains
        if allow_external:
            payload["allow_external"] = allow_external

        return payload

    def _make_api_request(self, api_key: str, payload: dict[str, Any]) -> str:
        """Make the API request and process the response."""
        api_url = "https://api.tavily.com/crawl"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

        try:
            response = httpx.post(api_url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            json_response = response.json()

            if "results" in json_response:
                return self._format_results(json_response["results"])

            self._raise_crawl_error(json_response)

        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
        except httpx.TimeoutException as e:
            raise ToolSoftError("Crawl request timed out") from e
        except Exception as e:
            raise ToolSoftError(f"Crawl request failed: {e!s}") from e

    def _format_results(self, results: list[Any]) -> str:
        """Format the crawl results into a readable string."""
        formatted_results = []
        for result in results:
            url_info = f"URL: {result.get('url', 'N/A')}"
            content_preview = result.get("raw_content", "")
            formatted_results.append(f"{url_info}\nContent: {content_preview}\n")

        return f"Crawled {len(results)} pages:\n\n" + "\n---\n".join(formatted_results)

    def _raise_crawl_error(self, json_response: dict[str, Any]) -> NoReturn:
        """Raise a ToolSoftError for crawl failures."""
        raise ToolSoftError(f"Failed to crawl website: {json_response}")

    def _handle_http_error(self, e: httpx.HTTPStatusError) -> NoReturn:
        """Handle HTTP errors with detailed error information."""
        error_detail = f"HTTP {e.response.status_code}"
        try:
            error_body = e.response.json()
            error_detail += f": {error_body}"
        except Exception:  # noqa: BLE001
            error_detail += f": {e.response.text}"
        raise ToolSoftError(f"Crawl API error - {error_detail}") from e
