"""Simple Search Tool."""

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel, Field

from portia.errors import ToolHardError, ToolSoftError
from portia.tool import Tool, ToolRunContext

MAX_RESULTS = 3


class SearchToolSchema(BaseModel):
    """Input for SearchTool."""

    search_query: str = Field(
        ...,
        description=(
            "The query to search for. For example, 'what is the capital of France?' or "
            "'who won the US election in 2020?'"
        ),
    )


class SearchTool(Tool[str]):
    """Searches the internet to find answers to the search query provided.."""

    id: str = "search_tool"
    name: str = "Search Tool"
    description: str = (
        "Searches the internet (using Tavily) to find answers to the search query provided and "
        "returns those answers, including images, links and a natural language answer. "
        "The search tool has access to general information but can not return specific "
        "information on users or information not available on the internet"
    )
    args_schema: type[BaseModel] = SearchToolSchema
    output_schema: tuple[str, str] = ("str", "str: output of the search results")
    should_summarize: bool = True

    def run(self, _: ToolRunContext, search_query: str) -> str:
        """Run the Search Tool."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key or api_key == "":
            raise ToolHardError("TAVILY_API_KEY is required to use search")

        url = "https://api.tavily.com/search"

        payload = {
            "query": search_query,
            "include_answer": True,
            "api_key": api_key,
        }
        headers = {"Content-Type": "application/json"}

        response = httpx.post(url, headers=headers, json=payload)
        response.raise_for_status()
        json_response = response.json()
        if "answer" in json_response:
            results = json_response["results"]
            return results[:MAX_RESULTS]
        raise ToolSoftError(f"Failed to get answer to search: {json_response}")
