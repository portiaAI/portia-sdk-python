"""OpenAI Search Tool."""

from __future__ import annotations

import json
import os
import re
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
    model: str = "gpt-5-mini"
    web_search_context_size: str = "medium"
    api_key: str | None = None

    def _get_api_key(self) -> str:
        """Get the OpenAI API key from instance or environment."""
        api_key_value = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key_value:
            raise ToolHardError("OPENAI_API_KEY is required to use OpenAI search")
        return api_key_value

    def _get_client(self) -> OpenAI:
        """Get or create the OpenAI client."""
        return OpenAI(api_key=self._get_api_key())

    def _get_async_client(self) -> AsyncOpenAI:
        """Get or create the async OpenAI client."""
        return AsyncOpenAI(api_key=self._get_api_key())

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
            client = self._get_client()
            response = client.responses.create(
                model=self.model,
                input=(
                    f"Search the web for '{search_query}' and provide results as a JSON array "
                    f"where each result has url, title, and content fields. Format: "
                    f'[{{"url": "...", "title": "...", "content": "..."}}, ...]'
                ),
                tools=[
                    {  # type: ignore[typeddict-item]
                        "type": "web_search_preview",
                        "web_search_preview": {"context_size": self.web_search_context_size},
                    }
                ],
                store=False,  # Don't store responses for privacy
                timeout=60.0,
            )
            return self._parse_formatted_response(response)
        except (ToolHardError, ToolSoftError):
            # Re-raise our own error types without modification
            raise
        except Exception as e:  # noqa: BLE001 - Need to catch all OpenAI API exceptions
            return self._handle_api_error(e)

    async def _execute_search_async(self, search_query: str) -> list[dict[str, Any]]:
        """Execute asynchronous search with proper error handling."""
        try:
            client = self._get_async_client()
            response = await client.responses.create(
                model=self.model,
                input=(
                    f"Search the web for '{search_query}' and provide results as a JSON array "
                    f"where each result has url, title, and content fields. Format: "
                    f'[{{"url": "...", "title": "...", "content": "..."}}, ...]'
                ),
                tools=[
                    {  # type: ignore[typeddict-item]
                        "type": "web_search_preview",
                        "web_search_preview": {"context_size": self.web_search_context_size},
                    }
                ],
                store=False,  # Don't store responses for privacy
                timeout=60.0,
            )
            return self._parse_formatted_response(response)
        except (ToolHardError, ToolSoftError):
            # Re-raise our own error types without modification
            raise
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

    def _parse_formatted_response(self, response: Any) -> list[dict[str, Any]]:  # noqa: ANN401
        """Parse the JSON response from OpenAI Response API."""
        if not response.output:
            return []

        try:
            # OpenAI Response API returns text in response.output.text
            response_text = response.output.text

            # Try to parse as JSON array directly
            if response_text.strip().startswith("["):
                return json.loads(response_text.strip())

            # If not direct JSON array, try to extract JSON from response text
            # Look for JSON array pattern in the response
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            # Fallback: try to parse entire response as JSON
            parsed_response = json.loads(response_text)
            if isinstance(parsed_response, list):
                return parsed_response
            if isinstance(parsed_response, dict) and "results" in parsed_response:
                return parsed_response["results"]
        except (json.JSONDecodeError, AttributeError, KeyError):
            # Return empty results if parsing fails - don't raise errors for empty results
            pass

        return []
