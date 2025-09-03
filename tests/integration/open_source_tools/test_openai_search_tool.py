"""Integration tests for the OpenAI search tool."""

from __future__ import annotations

import os
import pytest

from portia import LLMProvider, PlanRunState, Portia, ToolRegistry
from portia.config import Config, StorageClass
from portia.open_source_tools.openai_search_tool import OpenAISearchTool


@pytest.mark.daily
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_search_tool_integration() -> None:
    """Test that OpenAI search tool works end-to-end."""
    config = Config.from_default(
        llm_provider=LLMProvider.ANTHROPIC,
        storage_class=StorageClass.MEMORY,
    )

    # Create tool registry with OpenAI search tool
    tool_registry = ToolRegistry([OpenAISearchTool()])
    portia = Portia(config=config, tools=tool_registry)
    
    # Test search query
    query = "What is the capital of France?"

    plan_run = portia.run(query)
    
    # Verify the run completed successfully
    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output
    
    final_output = plan_run.outputs.final_output.get_value()
    assert "Paris" in final_output
    
    # Verify step outputs exist
    for output in plan_run.outputs.step_outputs.values():
        assert output.get_summary() is not None