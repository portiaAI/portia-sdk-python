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
    api_url: str = "https://api.tavily.com/search"

    def run(self, _: ToolRunContext, search_query: str) -> str:
        """Run the Search Tool."""
        payload, headers = self._prep_request(search_query)
        response = httpx.post(self.api_url, headers=headers, json=payload)
        return self._parse_response(response)

    async def arun(self, _: ToolRunContext, search_query: str) -> str:
        """Run the Search Tool asynchronously."""
        payload, headers = self._prep_request(search_query)
        async with httpx.AsyncClient() as client:
            response = await client.post(self.api_url, headers=headers, json=payload)
        return self._parse_response(response)

    def _check_valid_api_key(self) -> str:
        """Check if the API key is valid."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key or api_key == "":
            raise ToolHardError("TAVILY_API_KEY is required to use search")
        return api_key

    def _build_payload(self, search_query: str) -> dict:
        """Build the payload for the Search Tool."""
        return {
            "query": search_query,
            "include_answer": True,
        }

    def _build_headers(self, api_key: str) -> dict:
        """Build the headers for the Search Tool."""
        return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def _prep_request(self, search_query: str) -> tuple[dict, dict]:
        """Prepare the request for the Search Tool."""
        api_key = self._check_valid_api_key()
        payload = self._build_payload(search_query)
        headers = self._build_headers(api_key)
        return payload, headers

    def _parse_response(self, response: httpx.Response) -> str:
        """Parse the response from the Search Tool."""
        response.raise_for_status()
        json_response = response.json()
        if "answer" in json_response:
            results = json_response["results"]
            return results[:MAX_RESULTS]
        raise ToolSoftError(f"Failed to get answer to search: {json_response}")
