"""OpenAI Search Tool."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from pydantic import BaseModel, Field

from portia.errors import ToolHardError, ToolSoftError
from portia.tool import Tool, ToolRunContext

MAX_RESULTS = 3


class OpenAISearchToolSchema(BaseModel):
    """Input for OpenAISearchTool."""

    search_query: str = Field(
        ...,
        description=(
            "The query to search for. For example, 'what is the capital of France?' or "
            "'who won the US election in 2020?'"
        ),
    )


class OpenAISearchTool(Tool[str]):
    """Searches the internet using OpenAI's web search feature.
    
    This tool uses OpenAI's web search capability to find answers to search queries
    and returns the results in a format compatible with Tavily's search tool.
    """

    id: str = "openai_search_tool"
    name: str = "OpenAI Search Tool"
    description: str = (
        "Searches the internet (using OpenAI web search) to find answers to the search query provided and "
        "returns those answers, including links and a natural language answer. "
        "The search tool has access to general information but can not return specific "
        "information on users or information not available on the internet"
    )
    args_schema: type[BaseModel] = OpenAISearchToolSchema
    output_schema: tuple[str, str] = ("str", "str: output of the search results")
    should_summarize: bool = True
    api_url: str = "https://api.openai.com/v1/chat/completions"

    def run(self, _: ToolRunContext, search_query: str) -> str:
        """Run the OpenAI Search Tool."""
        payload, headers = self._prep_request(search_query)
        response = httpx.post(self.api_url, headers=headers, json=payload, timeout=60.0)
        return self._parse_response(response)

    async def arun(self, _: ToolRunContext, search_query: str) -> str:
        """Run the OpenAI Search Tool asynchronously."""
        payload, headers = self._prep_request(search_query)
        async with httpx.AsyncClient() as client:
            response = await client.post(self.api_url, headers=headers, json=payload, timeout=60.0)
        return self._parse_response(response)

    def _check_valid_api_key(self) -> str:
        """Check if the API key is valid."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "":
            raise ToolHardError("OPENAI_API_KEY is required to use OpenAI search")
        return api_key

    def _build_payload(self, search_query: str) -> dict[str, Any]:
        """Build the payload for the OpenAI Search Tool."""
        return {
            "model": "gpt-4o-search-preview",
            "web_search_options": {
                "search_context_size": "medium"
            },
            "messages": [
                {
                    "role": "user",
                    "content": f"search the web using search term '{search_query}' and provide all results"
                }
            ]
        }

    def _build_headers(self, api_key: str) -> dict[str, str]:
        """Build the headers for the OpenAI Search Tool."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def _prep_request(self, search_query: str) -> tuple[dict[str, Any], dict[str, str]]:
        """Prepare the request for the OpenAI Search Tool."""
        api_key = self._check_valid_api_key()
        payload = self._build_payload(search_query)
        headers = self._build_headers(api_key)
        return payload, headers

    def _parse_response(self, response: httpx.Response) -> str:
        """Parse the response from the OpenAI Search Tool."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ToolSoftError(f"OpenAI API error: {e.response.status_code} - {e.response.text}") from e
        
        try:
            json_response = response.json()
        except json.JSONDecodeError as e:
            raise ToolSoftError(f"Failed to parse OpenAI response: {e}") from e

        if "choices" not in json_response or not json_response["choices"]:
            raise ToolSoftError(f"No choices in OpenAI response: {json_response}")

        choice = json_response["choices"][0]
        message = choice.get("message", {})
        
        # Extract URLs from annotations to mimic Tavily's results format
        annotations = message.get("annotations", [])
        results = []
        
        for annotation in annotations:
            if annotation.get("type") == "url_citation":
                url_citation = annotation.get("url_citation", {})
                url = url_citation.get("url")
                title = url_citation.get("title", "")
                
                if url:
                    # Create a result object similar to what Tavily returns
                    result = {
                        "url": url,
                        "title": title,
                        "content": message.get("content", "")[:200] + "..." if len(message.get("content", "")) > 200 else message.get("content", "")
                    }
                    results.append(result)
        
        # If no annotations, try to extract content and create a single result
        if not results:
            content = message.get("content", "")
            if content:
                # Create a basic result with the content
                result = {
                    "content": content,
                    "title": "Search Results",
                    "url": ""
                }
                results.append(result)
        
        # Return first MAX_RESULTS results to match Tavily format
        limited_results = results[:MAX_RESULTS]
        
        if not limited_results:
            raise ToolSoftError(f"No search results found in OpenAI response: {json_response}")
        
        return limited_results