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

    This tool uses OpenAI's Response API with web search capability to find answers to
    search queries and returns the results in a format compatible with Tavily's search tool.
    """

    id: str = "openai_search_tool"
    name: str = "OpenAI Search Tool"
    description: str = (
        "Searches the internet (using OpenAI web search) to find answers to the search query "
        "provided and returns those answers, including links and a natural language answer. "
        "The search tool has access to general information but can not return specific "
        "information on users or information not available on the internet"
    )
    args_schema: type[BaseModel] = OpenAISearchToolSchema
    output_schema: tuple[str, str] = ("list", "list: search results with urls, titles, and content")
    should_summarize: bool = True

    def __init__(
        self,
        model: str = "gpt-5-mini",
        web_search_context_size: str = "medium",
        api_key: str | None = None,
    ) -> None:
        """Initialize the OpenAI Search Tool.

        Args:
            model: The model to use for search (default: gpt-5-mini)
            web_search_context_size: Context size for web search ("small", "medium", "large")
            api_key: OpenAI API key, if None will use OPENAI_API_KEY env var

        """
        super().__init__()
        self._model = model
        self._web_search_context_size = web_search_context_size

        api_key_value = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key_value:
            raise ToolHardError("OPENAI_API_KEY is required to use OpenAI search")

        self._client = OpenAI(api_key=api_key_value)
        self._async_client = AsyncOpenAI(api_key=api_key_value)

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
        return self._execute_search_sync(search_query)

    async def arun(self, _: ToolRunContext, search_query: str) -> list[dict[str, Any]]:
        """Run the OpenAI Search Tool asynchronously."""
        return await self._execute_search_async(search_query)

    def _get_response_format(self) -> dict[str, Any]:
        """Get the response format schema for structured output."""
        return {
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
                                    "content": {"type": "string"},
                                },
                                "required": ["url", "title", "content"],
                            },
                        }
                    },
                    "required": ["results"],
                },
            },
        }

    def _execute_search_sync(self, search_query: str) -> list[dict[str, Any]]:
        """Execute synchronous search with proper error handling."""
        try:
            response = self._client.responses.create(
                model=self._model,
                input=(
                    f"search the web using search term '{search_query}' and provide all results "
                    f"in the specified JSON format with url, title, and content for each result"
                ),
                response_format=self._get_response_format(),
                tools=[
                    {
                        "type": "web_search_preview",
                        "web_search_preview": {"context_size": self._web_search_context_size},
                    }
                ],
                store=False,  # Don't store responses for privacy
                timeout=60.0,
            )
            return self._parse_formatted_response(response)
        except Exception as e:  # noqa: BLE001 - Need to catch all OpenAI API exceptions
            return self._handle_api_error(e)

    async def _execute_search_async(self, search_query: str) -> list[dict[str, Any]]:
        """Execute asynchronous search with proper error handling."""
        try:
            response = await self._async_client.responses.create(
                model=self._model,
                input=(
                    f"search the web using search term '{search_query}' and provide all results "
                    f"in the specified JSON format with url, title, and content for each result"
                ),
                response_format=self._get_response_format(),
                tools=[
                    {
                        "type": "web_search_preview",
                        "web_search_preview": {"context_size": self._web_search_context_size},
                    }
                ],
                store=False,  # Don't store responses for privacy
                timeout=60.0,
            )
            return self._parse_formatted_response(response)
        except Exception as e:  # noqa: BLE001 - Need to catch all OpenAI API exceptions
            return self._handle_api_error(e)

    def _handle_api_error(self, e: Exception) -> list[dict[str, Any]]:
        """Handle API errors with appropriate exception types."""
        if "401" in str(e) or "unauthorized" in str(e).lower():
            raise ToolHardError(f"Invalid OpenAI API key: {e}") from e
        if "429" in str(e) or "rate limit" in str(e).lower():
            raise ToolSoftError(f"OpenAI API rate limit exceeded: {e}") from e
        if "500" in str(e) or "502" in str(e) or "503" in str(e) or "504" in str(e):
            raise ToolSoftError(f"OpenAI API server error: {e}") from e
        raise ToolSoftError(f"OpenAI API error: {e}") from e

    def _parse_formatted_response(self, response: object) -> list[dict[str, Any]]:
        """Parse the structured JSON response from OpenAI Response API."""
        if not hasattr(response, "output") or not response.output:
            return []

        try:
            # OpenAI Response API with JSON schema returns structured output consistently
            response_text = response.output.text
            parsed_response = json.loads(response_text)
            results = parsed_response.get("results", [])
            return results if isinstance(results, list) else []
        except (json.JSONDecodeError, AttributeError, KeyError):
            # Return empty results if parsing fails - don't raise errors for empty results
            return []
