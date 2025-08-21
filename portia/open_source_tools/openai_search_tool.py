"""OpenAI Search Tool."""

from __future__ import annotations

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
    
    This tool uses OpenAI's Response API with web search capability to find answers to search queries
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
        model: str = "gpt-5-mini",
        web_search_context_size: str = "medium",
        api_key: str | None = None,
    ):
        """Initialize the OpenAI Search Tool.
        
        Args:
            model: The model to use for search (default: gpt-5-mini)
            web_search_context_size: Context size for web search ("small", "medium", "large")
            api_key: OpenAI API key, if None will use OPENAI_API_KEY env var
        """
        super().__init__()
        self._model = model
        self._web_search_context_size = web_search_context_size
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self._api_key:
            raise ToolHardError("OPENAI_API_KEY is required to use OpenAI search")
            
        self._client = OpenAI(api_key=self._api_key)
        self._async_client = AsyncOpenAI(api_key=self._api_key)
    
    @property
    def model(self) -> str:
        """Get the model being used."""
        return self._model
    
    @property
    def web_search_context_size(self) -> str:
        """Get the web search context size."""
        return self._web_search_context_size
    
    @property
    def api_key(self) -> str:
        """Get the API key."""
        return self._api_key

    def run(self, _: ToolRunContext, search_query: str) -> list[dict[str, Any]]:
        """Run the OpenAI Search Tool."""
        try:
            response = self._client.responses.create(
                model=self._model,
                input=f"search the web using search term '{search_query}' and provide all results",
                tools=[
                    {
                        "type": "web_search_preview",
                        "web_search_preview": {
                            "context_size": self._web_search_context_size
                        }
                    }
                ],
                store=False,  # Don't store responses for privacy
                timeout=60.0
            )
        except Exception as e:
            if "401" in str(e) or "unauthorized" in str(e).lower():
                raise ToolHardError(f"Invalid OpenAI API key: {e}") from e
            elif "429" in str(e) or "rate limit" in str(e).lower():
                raise ToolSoftError(f"OpenAI API rate limit exceeded: {e}") from e
            elif "500" in str(e) or "502" in str(e) or "503" in str(e) or "504" in str(e):
                raise ToolSoftError(f"OpenAI API server error: {e}") from e
            else:
                raise ToolSoftError(f"OpenAI API error: {e}") from e
                
        return self._parse_response(response)

    async def arun(self, _: ToolRunContext, search_query: str) -> list[dict[str, Any]]:
        """Run the OpenAI Search Tool asynchronously."""
        try:
            response = await self._async_client.responses.create(
                model=self._model,
                input=f"search the web using search term '{search_query}' and provide all results",
                tools=[
                    {
                        "type": "web_search_preview",
                        "web_search_preview": {
                            "context_size": self._web_search_context_size
                        }
                    }
                ],
                store=False,  # Don't store responses for privacy
                timeout=60.0
            )
        except Exception as e:
            if "401" in str(e) or "unauthorized" in str(e).lower():
                raise ToolHardError(f"Invalid OpenAI API key: {e}") from e
            elif "429" in str(e) or "rate limit" in str(e).lower():
                raise ToolSoftError(f"OpenAI API rate limit exceeded: {e}") from e
            elif "500" in str(e) or "502" in str(e) or "503" in str(e) or "504" in str(e):
                raise ToolSoftError(f"OpenAI API server error: {e}") from e
            else:
                raise ToolSoftError(f"OpenAI API error: {e}") from e
                
        return self._parse_response(response)

    def _parse_response(self, response) -> list[dict[str, Any]]:
        """Parse the response from the OpenAI Response API."""
        if not hasattr(response, 'output') or not response.output:
            raise ToolSoftError(f"No output in OpenAI response: {response}")

        # Extract URLs from annotations to mimic Tavily's results format
        results = self._extract_search_results_from_annotations(response.output)
        
        # If no annotations, try to extract content and create a single result
        if not results:
            content = response.output_text or ""
            if content:
                # Create a basic result with the content
                result = {
                    "content": content,
                    "title": "Search Results",
                    "url": ""
                }
                results.append(result)
        
        if not results:
            raise ToolSoftError(f"No search results found in OpenAI response: {response}")
        
        return results

    def _extract_search_results_from_annotations(self, output) -> list[dict[str, Any]]:
        """Extract search results from response output annotations.
        
        Args:
            output: The output object from OpenAI response
            
        Returns:
            List of search result dictionaries with url, title, and content fields
        """
        results = []
        
        # Handle output as a list of content items
        if isinstance(output, list):
            for content_item in output:
                if hasattr(content_item, 'annotations'):
                    annotations = content_item.annotations or []
                    for annotation in annotations:
                        if getattr(annotation, 'type', None) == 'url_citation':
                            url_citation = getattr(annotation, 'url_citation', {})
                            url = getattr(url_citation, 'url', None)
                            title = getattr(url_citation, 'title', '') or ''
                            
                            if url:
                                result = {
                                    "url": url,
                                    "title": title,
                                    "content": getattr(content_item, 'text', '') or ''
                                }
                                results.append(result)
        
        # Handle single output item
        elif hasattr(output, 'annotations'):
            annotations = output.annotations or []
            for annotation in annotations:
                if getattr(annotation, 'type', None) == 'url_citation':
                    url_citation = getattr(annotation, 'url_citation', {})
                    url = getattr(url_citation, 'url', None)
                    title = getattr(url_citation, 'title', '') or ''
                    
                    if url:
                        result = {
                            "url": url,
                            "title": title,
                            "content": getattr(output, 'text', '') or ''
                        }
                        results.append(result)
        
        return results