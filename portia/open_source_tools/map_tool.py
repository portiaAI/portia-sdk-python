"""Tool to map websites."""

from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import BaseModel, Field

from portia.errors import ToolHardError, ToolSoftError
from portia.tool import Tool, ToolRunContext


class MapToolSchema(BaseModel):
    """Input for MapTool."""

    url: str = Field(..., description="The root URL to begin the mapping (e.g., 'docs.tavily.com')")
    max_depth: int = Field(
        default=1,
        description=(
            "Max depth of the mapping. Defines how far from the base URL the crawler can explore"
        ),
    )
    max_breadth: int = Field(
        default=20,
        description="Max number of links to follow per level of the tree (i.e., per page)",
    )
    limit: int = Field(
        default=50,
        description="Total number of links the crawler will process before stopping",
    )
    instructions: str | None = Field(
        default=None,
        description="Natural language instructions for the crawler (e.g., 'Python SDK')",
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
    categories: list[str] | None = Field(
        default=None,
        description=(
            "Filter URLs using predefined categories like 'Documentation', 'Blog', 'API', etc."
        ),
    )


class MapTool(Tool[str]):
    """Maps websites using Tavily's graph-based traversal to generate comprehensive site maps."""

    id: str = "map_tool"
    name: str = "Map Tool"
    description: str = (
        "Maps websites using graph-based traversal that can explore hundreds of paths "
        "in parallel with intelligent discovery to generate comprehensive site maps. "
        "Provide a URL and the tool will discover and return all accessible pages on that website. "
        "Supports depth control, domain filtering, path selection, and various mapping options "
        "for comprehensive site reconnaissance and URL discovery."
    )
    args_schema: type[BaseModel] = MapToolSchema
    output_schema: tuple[str, str] = ("str", "str: list of discovered URLs on the website")

    def run(
        self,
        _: ToolRunContext,
        url: str,
        max_depth: int = 1,
        max_breadth: int = 20,
        limit: int = 50,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Run the map tool."""
        return self._execute_map_request(
            url=url,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
            instructions=instructions,
            **kwargs,
        )

    def _execute_map_request(
        self,
        url: str,
        max_depth: int,
        max_breadth: int,
        limit: int,
        instructions: str | None,
        **kwargs: Any,
    ) -> str:
        """Execute the map request with the given parameters."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key or api_key == "":
            raise ToolHardError("TAVILY_API_KEY is required to use map")

        payload = self._build_payload(
            url=url,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
            instructions=instructions,
            **kwargs,
        )

        return self._make_api_request(api_key, payload)

    def _build_payload(
        self,
        url: str,
        max_depth: int,
        max_breadth: int,
        limit: int,
        instructions: str | None,
        **kwargs: Any,
    ) -> dict:
        """Build the API payload."""
        payload = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "allow_external": kwargs.get("allow_external", False),
        }

        if instructions is not None:
            payload["instructions"] = instructions

        optional_keys = [
            "select_paths",
            "select_domains",
            "exclude_paths",
            "exclude_domains",
            "categories",
        ]
        for key in optional_keys:
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        return payload

    def _make_api_request(self, api_key: str, payload: dict) -> str:
        """Make the API request."""
        api_url = "https://api.tavily.com/map"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

        response = httpx.post(api_url, headers=headers, json=payload, timeout=60.0)
        response.raise_for_status()
        json_response = response.json()

        if "results" in json_response:
            return json_response["results"]
        raise ToolSoftError(f"Failed to map website: {json_response}")
