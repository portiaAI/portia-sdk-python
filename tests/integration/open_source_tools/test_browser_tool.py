"""Integration tests for the browser tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from portia import LLMProvider, PlanBuilder, PlanRunState, Portia, ToolRegistry, ToolRunContext
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

if TYPE_CHECKING:
    from pydantic import BaseModel

    from portia.clarification import ActionClarification

STORAGE = [
    StorageClass.DISK,
    StorageClass.MEMORY,
    StorageClass.CLOUD,
]


@pytest.mark.daily
@pytest.mark.flaky(reruns=3)
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
        "Go to the Portia website (https://www.portialabs.ai) and retrieve the title of the first "
        "section of the homepage, it should contain the word 'environments'."
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
@pytest.mark.flaky(reruns=3)
def test_browser_tool_allowed_domains_blocks_invalid_url() -> None:
    """Test that browser tool with allowed_domains blocks navigation to invalid domains."""
    config = Config.from_default(
        llm_provider=LLMProvider.ANTHROPIC,
        storage_class=StorageClass.MEMORY,
    )

    allowed_domains = ["portialabs.ai"]
    browser_tool = BrowserTool(
        infrastructure_option=BrowserInfrastructureOption.REMOTE,
        allowed_domains=allowed_domains,
    )
    
    tool_registry = ToolRegistry([browser_tool])
    portia = Portia(config=config, tools=tool_registry)
    
    query = (
        "Go to https://example.com and retrieve the page title"
    )

    plan_run = portia.run(query)
    
    final_output = plan_run.outputs.final_output
    if final_output:
        output_text = final_output.get_value().lower()
        assert "example domain" not in output_text
        assert "this domain is for use in illustrative examples" not in output_text
    
    assert plan_run.state in [PlanRunState.COMPLETE, PlanRunState.ERROR], \
           f"Unexpected plan state: {plan_run.state}"


@pytest.mark.daily
@pytest.mark.flaky(reruns=3)
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
        "BrowserTool to go to the website and retrieve the title of the first section of the "
        "homepage, it should contain the word 'environments'."
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
@pytest.mark.flaky(reruns=3)
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
            "Retrieve the title of the first section of the homepage, it should contain the "
            "word 'environments'",
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
    assert (
        "Build AI agents you can trust in regulated environments"
        in step_outputs["page_title"].get_value()
    )  # type: ignore reportOperatorIssue
    assert "Welcome" in step_outputs["blog_title"].get_value()  # type: ignore reportOperatorIssue


class TestBrowserTool(BrowserTool):
    """Test browser tool that tracks when arun is called."""

    # Use class variable to track calls
    arun_called: ClassVar[bool] = False

    @classmethod
    def reset_tracking(cls) -> None:
        """Reset the tracking flag."""
        cls.arun_called = False

    async def arun(
        self,
        ctx: ToolRunContext,
        url: str,
        task: str,
        task_data: list[Any] | str | None = None,
    ) -> str | BaseModel | ActionClarification:
        """Track that arun was called and then call the parent implementation."""
        TestBrowserTool.arun_called = True
        return await super().arun(ctx, url, task, task_data)


@pytest.mark.daily
@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_portia_arun_calls_browser_tool_arun() -> None:
    """Test that portia.arun calls browser_tool.arun method."""
    config = Config.from_default(
        llm_provider=LLMProvider.ANTHROPIC,
        storage_class=StorageClass.MEMORY,
    )

    TestBrowserTool.reset_tracking()

    browser_tool = TestBrowserTool(infrastructure_option=BrowserInfrastructureOption.REMOTE)

    tool_registry = ToolRegistry([browser_tool])
    portia = Portia(config=config, tools=tool_registry)

    query = (
        "Go to the Portia website (https://www.portialabs.ai) and retrieve the title of the first "
        "section of the homepage."
    )

    plan_run = await portia.arun(query)

    assert TestBrowserTool.arun_called, "browser_tool.arun method was not called"
    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output
    assert (
        "Build AI agents you can trust in regulated environments"
        in plan_run.outputs.final_output.get_value()
    )  # type: ignore reportOperatorIssue
    for output in plan_run.outputs.step_outputs.values():
        assert output.get_summary() is not None