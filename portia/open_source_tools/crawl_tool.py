"""Tool to crawl websites."""

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel, Field

from portia.errors import ToolHardError, ToolSoftError
from portia.tool import Tool, ToolRunContext


class CrawlToolSchema(BaseModel):
    """Input for CrawlTool."""

    url: str = Field(
        ..., description="The root URL to begin the crawl (e.g., 'https://docs.tavily.com')"
    )
    instructions: str | None = Field(
        default=None,
        description="Natural language instructions for the crawler (e.g., 'Find all pages on the Python SDK')",
    )
    max_depth: int = Field(
        default=1,
        description="Max depth of the crawl. Defines how far from the base URL the crawler can explore",
        ge=1,
        le=5,
    )
    max_breadth: int = Field(
        default=20,
        description="Max number of links to follow per level of the tree (i.e., per page)",
        ge=1,
        le=100,
    )
    limit: int = Field(
        default=50,
        description="Total number of links the crawler will process before stopping",
        ge=1,
        le=500,
    )
    select_paths: list[str] | None = Field(
        default=None,
        description="Regex patterns to select only URLs with specific path patterns (e.g., ['/docs/.*', '/api/v1.*'])",
    )
    select_domains: list[str] | None = Field(
        default=None,
        description="Regex patterns to select crawling to specific domains or subdomains (e.g., ['^docs\\.example\\.com$'])",
    )
    exclude_paths: list[str] | None = Field(
        default=None,
        description="Regex patterns to exclude URLs with specific path patterns (e.g., ['/private/.*', '/admin/.*'])",
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description="Regex patterns to exclude specific domains or subdomains from crawling (e.g., ['^private\\.example\\.com$'])",
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
        max_depth: int = 1,
        max_breadth: int = 20,
        limit: int = 50,
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

        api_url = "https://api.tavily.com/crawl"

        # Build minimal payload following the official API docs
        payload = {
            "url": url,
        }

        # Add optional parameters only when provided
        if instructions is not None:
            payload["instructions"] = instructions
        if max_depth != 1:  # Only include if different from default
            payload["max_depth"] = max_depth
        if max_breadth != 20:  # Only include if different from default
            payload["max_breadth"] = max_breadth
        if limit != 50:  # Only include if different from default
            payload["limit"] = limit
        if select_paths is not None:
            payload["select_paths"] = select_paths
        if select_domains is not None:
            payload["select_domains"] = select_domains
        if exclude_paths is not None:
            payload["exclude_paths"] = exclude_paths
        if exclude_domains is not None:
            payload["exclude_domains"] = exclude_domains
        if allow_external:  # Only include if True
            payload["allow_external"] = allow_external

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

        try:
            response = httpx.post(api_url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            json_response = response.json()

            if "results" in json_response:
                results = json_response["results"]
                # Format the results nicely
                formatted_results = []
                for result in results:
                    url_info = f"URL: {result.get('url', 'N/A')}"
                    content_preview = result.get("raw_content", "")
                    formatted_results.append(f"{url_info}\nContent: {content_preview}\n")

                return f"Crawled {len(results)} pages:\n\n" + "\n---\n".join(formatted_results)
            raise ToolSoftError(f"Failed to crawl website: {json_response}")

        except httpx.HTTPStatusError as e:
            error_detail = f"HTTP {e.response.status_code}"
            try:
                error_body = e.response.json()
                error_detail += f": {error_body}"
            except:
                error_detail += f": {e.response.text}"
            raise ToolSoftError(f"Crawl API error - {error_detail}")
        except httpx.TimeoutException:
            raise ToolSoftError("Crawl request timed out")
        except Exception as e:
            raise ToolSoftError(f"Crawl request failed: {e!s}")
