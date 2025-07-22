"""Tool to extract web page content from one or more URLs."""

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel, Field

from portia.errors import ToolHardError, ToolSoftError
from portia.tool import Tool, ToolRunContext


class ExtractToolSchema(BaseModel):
    """Input for ExtractTool."""

    urls: list[str] = Field(..., description="List of URLs to extract content from")
    include_images: bool = Field(
        default=False, description="Whether to include images in the extraction"
    )
    include_favicon: bool = Field(
        default=False, description="Whether to include favicon in the extraction"
    )
    extract_depth: str = Field(
        default="basic",
        description=(
            "The depth of the extraction process. Advanced extraction retrieves more data, "
            "including tables and embedded content, with higher success but may increase latency. "
            "Basic extraction costs 1 credit per 5 successful URL extractions, while advanced "
            "extraction costs 2 credits per 5 successful URL extractions."
        ),
    )
    format: str = Field(default="markdown", description="Output format: 'markdown' or 'text'")


class ExtractTool(Tool[str]):
    """Extracts the web page content from one or more URLs provided."""

    id: str = "extract_tool"
    name: str = "Extract Tool"
    description: str = (
        "Extracts web page content from one or more specified URLs using Tavily Extract and "
        "returns the raw content, images, and metadata from those pages. "
        "The extract tool can access publicly available web pages but cannot extract content "
        "from pages that block automated access"
    )
    args_schema: type[BaseModel] = ExtractToolSchema
    output_schema: tuple[str, str] = ("str", "str: extracted content from URLs")

    def run(
        self,
        _: ToolRunContext,
        urls: list[str],
        include_images: bool = True,
        include_favicon: bool = True,
        extract_depth: str = "basic",
        format: str = "markdown",  # noqa: A002, API requires 'format' field name
    ) -> str:
        """Run the extract tool."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key or api_key == "":
            raise ToolHardError("TAVILY_API_KEY is required to use extract")

        url = "https://api.tavily.com/extract"

        payload = {
            "urls": urls,
            "include_images": include_images,
            "include_favicon": include_favicon,
            "extract_depth": extract_depth,
            "format": format,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

        response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
        response.raise_for_status()
        json_response = response.json()

        if "results" in json_response:
            return json_response["results"]
        raise ToolSoftError(f"Failed to extract content: {json_response}")
