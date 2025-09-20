"""Integration tests for BrowserTool allowed_domains functionality.

This module contains integration tests that verify the allowed_domains feature
works correctly across the entire BrowserTool system.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from portia.config import Config
from portia.end_user import EndUser
from portia.open_source_tools.browser_tool import (
    BrowserInfrastructureOption,
    BrowserInfrastructureProviderLocal,
    BrowserTool,
    BrowserToolForUrl,
)
from portia.plan import Plan
from portia.plan_run import PlanRun
from portia.tool import ToolRunContext


class TestOutputModel(BaseModel):
    """Test output model for integration tests."""

    task_output: str


class TestBrowserToolAllowedDomainsIntegration:
    """Integration tests for the allowed_domains feature."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_end_user = Mock(spec=EndUser)
        self.mock_plan = Mock(spec=Plan)
        self.mock_plan_run = Mock(spec=PlanRun)

        self.mock_ctx = Mock(spec=ToolRunContext)
        self.mock_ctx.config = self.mock_config
        self.mock_ctx.end_user = self.mock_end_user
        self.mock_ctx.plan = self.mock_plan
        self.mock_ctx.plan_run = self.mock_plan_run

    def test_browser_tool_initialization_with_allowed_domains(self) -> None:
        """Test that BrowserTool can be initialized with allowed_domains."""
        allowed_domains = ["example.com", "trusted-site.org"]

        tool = BrowserTool(
            allowed_domains=allowed_domains,
            infrastructure_option=BrowserInfrastructureOption.LOCAL
        )

        assert tool.allowed_domains == allowed_domains
        assert tool.id == "browser_tool"
        assert tool.name == "Browser Tool"

    def test_browser_tool_for_url_initialization_with_allowed_domains(self) -> None:
        """Test that BrowserToolForUrl can be initialized with allowed_domains."""
        url = "https://example.com"
        allowed_domains = ["example.com"]

        tool = BrowserToolForUrl(
            url=url,
            infrastructure_option=BrowserInfrastructureOption.LOCAL
        )
        # Set allowed_domains after initialization since it's not in the constructor
        tool.allowed_domains = allowed_domains

        assert tool.url == url
        assert tool.allowed_domains == allowed_domains
        assert "example_com" in tool.id

    def test_allowed_domains_validation_integration(self) -> None:
        """Test that the field validation works during tool initialization."""
        # Test valid domains
        valid_domains = ["example.com", "subdomain.example.com"]
        tool = BrowserTool(allowed_domains=valid_domains)
        assert tool.allowed_domains == ["example.com", "subdomain.example.com"]

        # Test None (should work)
        tool = BrowserTool(allowed_domains=None)
        assert tool.allowed_domains is None

        # Test invalid input type
        with pytest.raises(ValueError, match="must be a list"):
            BrowserTool(allowed_domains="not a list")  # type: ignore[arg-type][arg-type][arg-type][arg-type][arg-type][arg-type]

        # Test empty domain
        with pytest.raises(ValueError, match="Invalid domain value"):
            BrowserTool(allowed_domains=[""])

    def test_whitespace_and_case_normalization(self) -> None:
        """Test that domains are properly normalized."""
        domains_with_issues = [" Example.COM ", "  SUBDOMAIN.example.com  "]
        tool = BrowserTool(allowed_domains=domains_with_issues)

        expected = ["example.com", "subdomain.example.com"]
        assert tool.allowed_domains == expected

    @patch("portia.open_source_tools.browser_tool.logger")
    def test_wildcard_warnings_integration(self, mock_logger: MagicMock) -> None:
        """Test that wildcard patterns generate appropriate warnings."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        # Test universal wildcard warning
        BrowserTool(allowed_domains=["*"])
        mock_logger_instance.warning.assert_called_with(
            "Universal wildcard '*' allows access to ANY domain. "
            "This is extremely dangerous. Use specific domain patterns instead."
        )

        # Reset mock
        mock_logger_instance.reset_mock()

        # Test pattern wildcard warning
        BrowserTool(allowed_domains=["*.example.com"])
        mock_logger_instance.warning.assert_called_with(
            "Wildcard pattern '*.example.com' may match unintended domains. "
            "Per browser-use docs, be very cautious with wildcards."
        )

    @patch("portia.open_source_tools.browser_tool.Browser")
    @patch("portia.open_source_tools.browser_tool.Agent")
    def test_allowed_domains_passed_to_infrastructure_provider(
        self, mock_agent: MagicMock, mock_browser: MagicMock  # noqa: ARG002
    ) -> None:
        """Test that allowed_domains are passed through to the infrastructure provider."""
        allowed_domains = ["example.com", "trusted-site.org"]

        tool = BrowserTool(
            allowed_domains=allowed_domains,
            infrastructure_option=BrowserInfrastructureOption.LOCAL
        )

        # Mock the infrastructure provider's setup_browser method
        mock_infrastructure = Mock()
        tool.custom_infrastructure_provider = mock_infrastructure

        # Mock other dependencies
        mock_model = Mock()
        mock_model.to_langchain.return_value = Mock()
        self.mock_config.get_generative_model.return_value = mock_model
        self.mock_config.get_default_model.return_value = mock_model

        mock_agent_instance = Mock()
        mock_agent_instance.run.return_value = Mock()
        success_result = '{"task_output": "success"}'
        mock_agent_instance.run.return_value.final_result.return_value = success_result
        mock_agent.return_value = mock_agent_instance

        # Test async method
        import asyncio
        async def test_async() -> None:
            await tool._run_agent_task(
                self.mock_ctx,
                "test task",
                TestOutputModel
            )

            # Verify setup_browser was called with allowed_domains
            mock_infrastructure.setup_browser.assert_called_once_with(
                self.mock_ctx,
                allowed_domains
            )

        # Run the async test
        asyncio.run(test_async())

    def test_infrastructure_provider_local_with_allowed_domains(self) -> None:
        """Test that local infrastructure provider handles allowed_domains correctly."""
        provider = BrowserInfrastructureProviderLocal()
        allowed_domains = ["example.com", "test.org"]

        # Mock Browser, BrowserConfig, and BrowserContextConfig
        with (
            patch("portia.open_source_tools.browser_tool.Browser") as mock_browser,
            patch("portia.open_source_tools.browser_tool.BrowserConfig") as mock_config,
            patch(
                "portia.open_source_tools.browser_tool.BrowserContextConfig"
            ) as mock_context_config,
        ):

            mock_config_instance = Mock()
            mock_config.return_value = mock_config_instance
            mock_context_config_instance = Mock()
            mock_context_config.return_value = mock_context_config_instance

            provider.setup_browser(self.mock_ctx, allowed_domains)

            # Verify that BrowserConfig was created
            mock_config.assert_called_once()

            # Verify that BrowserContextConfig was created with allowed_domains
            mock_context_config.assert_called_once_with(allowed_domains=allowed_domains)

            # Verify that new_context_config was set on the config
            assert mock_config_instance.new_context_config == mock_context_config_instance

            # Verify Browser was created with the config
            mock_browser.assert_called_once_with(config=mock_config_instance)

    def test_backward_compatibility(self) -> None:
        """Test that existing code without allowed_domains continues to work."""
        # Test default initialization
        tool = BrowserTool()
        assert tool.allowed_domains is None

        # Test BrowserToolForUrl
        tool_for_url = BrowserToolForUrl("https://example.com")
        assert tool_for_url.allowed_domains is None

        # Test that infrastructure providers work with None
        provider = BrowserInfrastructureProviderLocal()

        with (
            patch("portia.open_source_tools.browser_tool.Browser"),
            patch("portia.open_source_tools.browser_tool.BrowserConfig") as mock_config,
            patch(
                "portia.open_source_tools.browser_tool.BrowserContextConfig"
            ) as mock_context_config,
        ):

            mock_config_instance = Mock()
            mock_config.return_value = mock_config_instance

            provider.setup_browser(self.mock_ctx, None)

            # Verify that BrowserConfig was created
            mock_config.assert_called_once()

            # Verify that BrowserContextConfig was NOT created (since allowed_domains is None)
            mock_context_config.assert_not_called()

            # Verify that new_context_config was not set
            assert not hasattr(mock_config_instance, "new_context_config") or \
                   mock_config_instance.new_context_config is None

    def test_multiple_domain_types(self) -> None:
        """Test various domain formats are handled correctly."""
        domains = [
            "example.com",
            "subdomain.example.com",
            "another-site.org",
            "site-with-dashes.net",
            "numeric123.com"
        ]

        tool = BrowserTool(allowed_domains=domains)

        # All domains should be preserved and normalized
        expected = [d.lower() for d in domains]
        assert tool.allowed_domains == expected

    def test_tool_run_context_integration(self) -> None:
        """Test that the tool works correctly within a full ToolRunContext."""
        allowed_domains = ["example.com"]
        tool = BrowserTool(
            allowed_domains=allowed_domains,
            infrastructure_option=BrowserInfrastructureOption.LOCAL
        )

        # Verify the tool maintains its configuration
        assert tool.allowed_domains == allowed_domains
        assert isinstance(tool.infrastructure_provider, BrowserInfrastructureProviderLocal)

        # Verify args_schema includes the new field
        assert hasattr(tool, "allowed_domains")
