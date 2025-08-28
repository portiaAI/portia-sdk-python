"""OpenAI Search Tool."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

from portia.errors import ToolHardError, ToolSoftError
from portia.tool import Tool, ToolRunContext


class OpenAISearchToolSchema(BaseModel):
    """Input for OpenAISearchTool."""

    search_query: str = Field(
        ...,
        description=(
            "The query to search for. For example, 'what is the capital of France?' or "
            "'who won the US election in 2020?'"
        ),
    )


class OpenAISearchTool(Tool[list[dict[str, Any]]]):
    """Searches the internet using OpenAI's web search feature.
    
    This tool uses OpenAI's chat API with web search capability to find answers to search queries
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
    output_schema: tuple[str, str] = ("list", "list: search results with urls, titles, and content")
    should_summarize: bool = True

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        web_search_context_size: str = "medium",
    ):
        """Initialize the OpenAI Search Tool.
        
        Args:
            model: The model to use for search (default: gpt-4o-mini)
            web_search_context_size: Context size for web search ("small", "medium", "large")
        """
        super().__init__()
        self._model = model
        self._web_search_context_size = web_search_context_size
    
    @property
    def model(self) -> str:
        """Get the model being used."""
        return self._model
    
    @property
    def web_search_context_size(self) -> str:
        """Get the web search context size."""
        return self._web_search_context_size

    def run(self, _: ToolRunContext, search_query: str) -> list[dict[str, Any]]:
        """Run the OpenAI Search Tool."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ToolHardError("OPENAI_API_KEY is required to use OpenAI search")
        
        client = OpenAI(api_key=api_key)
        return self._search_with_client(client, search_query)

    async def arun(self, _: ToolRunContext, search_query: str) -> list[dict[str, Any]]:
        """Run the OpenAI Search Tool asynchronously."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ToolHardError("OPENAI_API_KEY is required to use OpenAI search")
        
        async_client = AsyncOpenAI(api_key=api_key)
        return await self._search_with_async_client(async_client, search_query)

    def _search_with_client(self, client: OpenAI, search_query: str) -> list[dict[str, Any]]:
        """Perform search using sync OpenAI client."""
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user", 
                        "content": f"Search the web for information about: {search_query}. Provide search results with URLs, titles, and content snippets."
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "search_results",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "results": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "url": {"type": "string"},
                                            "title": {"type": "string"},
                                            "content": {"type": "string"}
                                        },
                                        "required": ["url", "title", "content"]
                                    }
                                }
                            },
                            "required": ["results"]
                        }
                    }
                },
                timeout=60.0
            )
        except Exception as e:
            self._handle_api_error(e)
            
        return self._parse_response(response)
    
    async def _search_with_async_client(self, async_client: AsyncOpenAI, search_query: str) -> list[dict[str, Any]]:
        """Perform search using async OpenAI client."""
        try:
            response = await async_client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user", 
                        "content": f"Search the web for information about: {search_query}. Provide search results with URLs, titles, and content snippets."
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "search_results",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "results": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "url": {"type": "string"},
                                            "title": {"type": "string"},
                                            "content": {"type": "string"}
                                        },
                                        "required": ["url", "title", "content"]
                                    }
                                }
                            },
                            "required": ["results"]
                        }
                    }
                },
                timeout=60.0
            )
        except Exception as e:
            self._handle_api_error(e)
            
        return self._parse_response(response)
    
    def _handle_api_error(self, e: Exception) -> None:
        """Handle and raise appropriate errors for API failures."""
        if "401" in str(e) or "unauthorized" in str(e).lower():
            raise ToolHardError(f"Invalid OpenAI API key: {e}") from e
        elif "429" in str(e) or "rate limit" in str(e).lower():
            raise ToolSoftError(f"OpenAI API rate limit exceeded: {e}") from e
        elif "500" in str(e) or "502" in str(e) or "503" in str(e) or "504" in str(e):
            raise ToolSoftError(f"OpenAI API server error: {e}") from e
        else:
            raise ToolSoftError(f"OpenAI API error: {e}") from e
    
    def _parse_response(self, response) -> list[dict[str, Any]]:
        """Parse the structured response from OpenAI."""
        try:
            content = response.choices[0].message.content
            if not content:
                raise ToolSoftError("No content in OpenAI response")
            
            parsed = json.loads(content)
            results = parsed.get("results", [])
            
            if not results:
                raise ToolSoftError("No search results found in response")
            
            return results
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise ToolSoftError(f"Failed to parse OpenAI response: {e}") from e