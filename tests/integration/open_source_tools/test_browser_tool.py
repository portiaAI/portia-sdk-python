"""Integration tests for the browser tool."""

from __future__ import annotations

import pytest

from portia import LLMProvider, PlanBuilder, PlanRunState, Portia, ToolRegistry
from portia.config import (
    Config,
    StorageClass,
)
from portia.open_source_tools.browser_tool import (
    BrowserInfrastructureOption,
    BrowserTool,
    BrowserToolForUrl,
)
from portia.open_source_tools.registry import open_source_tool_registry

STORAGE = [
    StorageClass.DISK,
    StorageClass.MEMORY,
    StorageClass.CLOUD,
]


@pytest.mark.daily
@pytest.mark.parametrize("storage", STORAGE)
def test_portia_run_query(
    storage: StorageClass,
) -> None:
    """Test running a simple query."""
    config = Config.from_default(
        llm_provider=LLMProvider.ANTHROPIC,
        storage_class=storage,
    )

    tool_registry = ToolRegistry(
        [BrowserTool(infrastructure_option=BrowserInfrastructureOption.REMOTE)]
    )
    portia = Portia(config=config, tools=tool_registry)
    query = (
        "Go to the Portia website (https://www.portialabs.ai) and retrieve the title of the "
        " homepage."
    )

    plan_run = portia.run(query)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output
    assert (
        "Build AI agents you can trust in regulated environments"
        in plan_run.outputs.final_output.get_value()
    )  # type: ignore reportOperatorIssue
    for output in plan_run.outputs.step_outputs.values():
        assert output.get_summary() is not None


@pytest.mark.daily
def test_portia_run_query_multi_step() -> None:
    """Test running a query that passes data to the browser tool."""
    config = Config.from_default(
        llm_provider=LLMProvider.ANTHROPIC,
        storage_class=StorageClass.MEMORY,
    )

    portia = Portia(
        config=config,
        tools=(
            open_source_tool_registry  # type: ignore reportOperatorIssue
            + ToolRegistry([BrowserTool(infrastructure_option=BrowserInfrastructureOption.REMOTE)])
        ),
    )  # type: ignore reportOperatorIssue
    query = (
        "Retrieve the website for Portia AI by using the Tavily search tool, and then use the "
        "BrowserTool to go to the website and retrieve the title of the homepage."
    )

    plan_run = portia.run(query)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output
    assert (
        "Build AI agents you can trust in regulated environments"
        in plan_run.outputs.final_output.get_value()
    )  # type: ignore reportOperatorIssue
    for output in plan_run.outputs.step_outputs.values():
        assert output.get_summary() is not None


@pytest.mark.daily
def test_portia_multi_step_from_plan() -> None:
    """Test running a query that requires the browser tool to be invoked multiple times."""
    config = Config.from_default(
        llm_provider=LLMProvider.ANTHROPIC,
        storage_class=StorageClass.MEMORY,
    )

    portia = Portia(
        config=config,
        tools=(
            open_source_tool_registry  # type: ignore reportOperatorIssue
            + ToolRegistry(
                [
                    BrowserToolForUrl(
                        infrastructure_option=BrowserInfrastructureOption.REMOTE,
                        url="https://www.portialabs.ai",
                    ),
                    BrowserToolForUrl(
                        infrastructure_option=BrowserInfrastructureOption.REMOTE,
                        url="https://blog.portialabs.ai",
                    ),
                ]
            )
        ),
    )

    plan = (
        PlanBuilder()
        .step(
            "Retrieve the title of the homepage",
            "browser_tool_for_url_www_portialabs_ai",
            "page_title",
        )
        .step("Find the price of gold", "search_tool", "search_result")
        .step(
            "Find the title of the first blog that was published chronologically",
            "browser_tool_for_url_blog_portialabs_ai",
            "blog_title",
        )
        .build()
    )

    plan_run = portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output
    step_outputs = plan_run.outputs.step_outputs
    assert "Build agents that are predictable" in step_outputs["page_title"].get_value()  # type: ignore reportOperatorIssue
    assert "Welcome" in step_outputs["blog_title"].get_value()  # type: ignore reportOperatorIssue
