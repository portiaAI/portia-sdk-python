"""Tests for the browser tool."""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from browser_use import Browser
from pydantic import BaseModel, Field, HttpUrl

from portia import ActionClarification, ToolHardError, ToolRunContext
from portia.end_user import EndUser
from portia.open_source_tools.browser_tool import (
    BrowserInfrastructureOption,
    BrowserInfrastructureProvider,
    BrowserInfrastructureProviderBrowserBase,
    BrowserInfrastructureProviderLocal,
    BrowserTaskOutput,
    BrowserTool,
    BrowserToolForUrl,
    BrowserToolForUrlSchema,
)
from portia.plan import PlanBuilder
from portia.plan_run import PlanRun
from portia.prefixed_uuid import PlanRunUUID
from tests.utils import assert_clarification_equality_without_uuid, get_test_tool_context


@pytest.fixture
def browser_tool() -> BrowserTool:
    """Return Browser Tool."""
    return BrowserTool()


@pytest.fixture
def mock_agent() -> MagicMock:
    """Return mock agent."""
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock()
    return mock_agent


@pytest.fixture
def local_browser_provider() -> BrowserInfrastructureProviderLocal:
    """Return a BrowserInfrastructureProviderLocal instance."""
    return BrowserInfrastructureProviderLocal()


@pytest.fixture
def mock_browserbase() -> MagicMock:
    """Return a mock Browserbase instance."""
    mock_bb = MagicMock()
    mock_bb.contexts = MagicMock()
    mock_bb.sessions = MagicMock()
    return mock_bb


@pytest.fixture
def mock_browserbase_provider(
    mock_browserbase: MagicMock,
) -> BrowserInfrastructureProviderBrowserBase:
    """Return a BrowserInfrastructureProviderBrowserBase with mocked Browserbase."""
    with patch.dict(
        os.environ,
        {
            "BROWSERBASE_API_KEY": "test_key",
            "BROWSERBASE_PROJECT_ID": "test_project",
        },
    ):
        provider = BrowserInfrastructureProviderBrowserBase()
        provider.bb = mock_browserbase
        return provider


class MockBrowserInfrastructureProvider(BrowserInfrastructureProvider):
    """Mock browser infrastructure provider."""

    def setup_browser(self, _: ToolRunContext) -> Browser:  # type: ignore reportIncompatibleMethodOverride
        """Create the browser with a mock for testing."""
        return MagicMock()

    def construct_auth_clarification_url(self, _: ToolRunContext, sign_in_url: str) -> HttpUrl:  # type: ignore reportIncompatibleMethodOverride
        """Construct the auth clarification for testing."""
        return HttpUrl(sign_in_url)

    def step_complete(self, _: ToolRunContext) -> None:  # type: ignore reportIncompatibleMethodOverride
        """Call when the step is complete to e.g release the session."""


@pytest.fixture
def mock_browser_infrastructure_provider() -> BrowserInfrastructureProvider:
    """Mock browser infrastructure provider."""
    return MockBrowserInfrastructureProvider()


def test_browser_tool_auth_check(
    mock_browser_infrastructure_provider: BrowserInfrastructureProvider,
) -> None:
    """Test the authentication check in browser tool."""
    # Mock response data
    mock_auth_response = BrowserTaskOutput(
        human_login_required=True,
        login_url="https://example.com/login",
        user_login_guidance="Login to example.com",
    )

    # Create a mock result object
    mock_result = MagicMock()
    mock_result.final_result.return_value = json.dumps(mock_auth_response.model_dump())

    # Create async mock for agent.run()
    mock_run = AsyncMock(return_value=mock_result)

    # Path the Agent class
    with patch("portia.open_source_tools.browser_tool.Agent") as mock_agent:
        # Configure the mock Agent instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = mock_run
        mock_agent.return_value = mock_agent_instance

        browser_tool = BrowserTool(
            custom_infrastructure_provider=mock_browser_infrastructure_provider,
        )
        context = get_test_tool_context()

        # Run the tool
        result = browser_tool.run(context, "https://example.com", "test task")

        # Verify Agent was called with correct parameters
        mock_agent.assert_called_once()
        mock_run.assert_called_once()

        assert isinstance(result, ActionClarification)
        assert_clarification_equality_without_uuid(
            result,
            ActionClarification(
                user_guidance="Login to example.com",
                action_url=HttpUrl("https://example.com/login"),
                plan_run_id=context.plan_run.id,
                require_confirmation=True,
                source="Browser tool",
            ),
        )


def test_browser_tool_bad_response(
    mock_browser_infrastructure_provider: BrowserInfrastructureProvider,
) -> None:
    """Test the browser tool when no authentication is required."""
    # Mock auth response data but with bad data
    mock_auth_response = BrowserTaskOutput(
        human_login_required=True,
    )
    # Create mock result objects for both auth check and task
    mock_auth_result = MagicMock()
    mock_auth_result.final_result.return_value = json.dumps(mock_auth_response.model_dump())

    # Create async mock for agent.run() that returns different results for auth and task
    mock_run = AsyncMock(return_value=mock_auth_result)

    # Patch the Agent class
    with patch("portia.open_source_tools.browser_tool.Agent") as mock_agent:
        # Configure the mock Agent instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = mock_run
        mock_agent.return_value = mock_agent_instance

        browser_tool = BrowserTool(
            custom_infrastructure_provider=mock_browser_infrastructure_provider
        )
        context = get_test_tool_context()

        # Run the tool
        with pytest.raises(ToolHardError, match="Expected user guidance and login URL"):
            browser_tool.run(context, "https://example.com", "test task")

        # Verify Agent was called once
        assert mock_agent.call_count == 1
        assert mock_run.call_count == 1


def test_browser_tool_no_auth_required(
    mock_browser_infrastructure_provider: BrowserInfrastructureProvider,
) -> None:
    """Test the browser tool when no authentication is required."""
    # Mock task response data
    mock_task_response = BrowserTaskOutput(
        task_output="Task completed successfully",
        human_login_required=False,
    )

    mock_task_result = MagicMock()
    mock_task_result.final_result.return_value = json.dumps(mock_task_response.model_dump())

    # Create async mock for agent.run() that returns different results for auth and task
    mock_run = AsyncMock(return_value=mock_task_result)

    # Patch the Agent class
    with patch("portia.open_source_tools.browser_tool.Agent") as mock_agent:
        # Configure the mock Agent instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = mock_run
        mock_agent.return_value = mock_agent_instance

        browser_tool = BrowserTool(
            custom_infrastructure_provider=mock_browser_infrastructure_provider
        )
        context = get_test_tool_context()

        # Run the tool
        result = browser_tool.run(context, "https://example.com", "test task")

        # Verify Agent was called once
        assert mock_agent.call_count == 1
        assert mock_run.call_count == 1

        # Verify the final result is the task output
        assert result == "Task completed successfully"


def test_browser_tool_infrastructure_provider_custom() -> None:
    """Test infrastructure_provider property returns custom provider when set."""
    mock_provider = MockBrowserInfrastructureProvider()
    browser_tool = BrowserTool(custom_infrastructure_provider=mock_provider)

    assert browser_tool.infrastructure_provider is mock_provider


def test_browser_tool_infrastructure_provider_remote() -> None:
    """Test infrastructure_provider property returns BrowserBase provider for REMOTE option."""
    browser_tool = BrowserTool(infrastructure_option=BrowserInfrastructureOption.REMOTE)

    with patch(
        "portia.open_source_tools.browser_tool.BrowserInfrastructureProviderBrowserBase"
    ) as mock_browserbase_provider_class:
        mock_provider = MagicMock()
        mock_browserbase_provider_class.return_value = mock_provider

        provider = browser_tool.infrastructure_provider

        assert provider is mock_provider
        mock_browserbase_provider_class.assert_called_once()


def test_browser_tool_infrastructure_provider_local() -> None:
    """Test infrastructure_provider property returns Local provider for LOCAL option."""
    browser_tool = BrowserTool(infrastructure_option=BrowserInfrastructureOption.LOCAL)

    with patch(
        "portia.open_source_tools.browser_tool.BrowserInfrastructureProviderLocal"
    ) as mock_local_provider_class:
        mock_provider = MagicMock()
        mock_local_provider_class.return_value = mock_provider

        provider = browser_tool.infrastructure_provider

        assert provider is mock_provider
        mock_local_provider_class.assert_called_once()


def test_browser_infra_local_get_chrome_instance_path_from_env(
    local_browser_provider: BrowserInfrastructureProviderLocal,
) -> None:
    """Test chrome path resolution from environment variable."""
    with patch.dict(os.environ, {"PORTIA_BROWSER_LOCAL_CHROME_EXEC": "/custom/chrome/path"}):
        assert local_browser_provider.get_chrome_instance_path() == "/custom/chrome/path"


def test_browser_infra_local_get_chrome_instance_path_by_platform(
    local_browser_provider: BrowserInfrastructureProviderLocal,
) -> None:
    """Test chrome path resolution based on platform."""
    platforms = {
        "darwin": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "win32": r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "linux": "/usr/bin/google-chrome",
    }

    for platform, expected_path in platforms.items():
        with patch("sys.platform", platform), patch.dict(os.environ, clear=True):
            local_browser_provider = BrowserInfrastructureProviderLocal()
            assert local_browser_provider.get_chrome_instance_path() == expected_path


def test_browser_infra_local_get_chrome_instance_path_unsupported_platform(
    local_browser_provider: BrowserInfrastructureProviderLocal,
) -> None:
    """Test chrome path resolution with unsupported platform."""
    with (
        patch.object(sys, "platform", "unsupported"),
        patch.dict(os.environ, clear=True),
        pytest.raises(RuntimeError, match="Unsupported platform: unsupported"),
    ):
        local_browser_provider.get_chrome_instance_path()


def test_browser_infra_local_get_extra_chromium_args_from_env(
    local_browser_provider: BrowserInfrastructureProviderLocal,
) -> None:
    """Test extra chromium args from environment variable."""
    test_args = "--headless,--disable-gpu"
    with patch.dict(os.environ, {"PORTIA_BROWSER_LOCAL_EXTRA_CHROMIUM_ARGS": test_args}):
        assert local_browser_provider.get_extra_chromium_args() == ["--headless", "--disable-gpu"]


def test_browser_infra_local_get_extra_chromium_args_default(
    local_browser_provider: BrowserInfrastructureProviderLocal,
) -> None:
    """Test default extra chromium args."""
    with patch.dict(os.environ, clear=True):
        assert local_browser_provider.get_extra_chromium_args() is None


def test_browser_infra_local_setup_browser(
    local_browser_provider: BrowserInfrastructureProviderLocal,
) -> None:
    """Test browser setup."""
    context = get_test_tool_context()
    context.end_user = EndUser(external_id="test_user")

    mock_logger_instance = MagicMock()
    mock_logger = MagicMock(return_value=mock_logger_instance)
    with patch("portia.open_source_tools.browser_tool.logger", mock_logger):
        browser = local_browser_provider.setup_browser(context)

        # Verify warning was logged for end_user
        mock_logger.assert_called_once()
        mock_logger_instance.warning.assert_called_once()
        assert "does not support end users" in mock_logger_instance.warning.call_args[0][0]

        # Verify browser instance
        assert isinstance(browser, Browser)


def test_browser_infra_local_construct_auth_clarification_url(
    local_browser_provider: BrowserInfrastructureProviderLocal,
) -> None:
    """Test auth clarification URL construction."""
    context = get_test_tool_context()
    sign_in_url = "https://example.com/login"

    result = local_browser_provider.construct_auth_clarification_url(context, sign_in_url)
    assert isinstance(result, HttpUrl)
    assert str(result) == sign_in_url


def test_browser_infra_local_custom_chrome_path_initialization() -> None:
    """Test initialization with custom chrome path."""
    custom_path = "/custom/chrome/path"
    provider = BrowserInfrastructureProviderLocal(chrome_path=custom_path)
    assert provider.chrome_path == custom_path


def test_browser_infra_local_custom_chromium_args_initialization() -> None:
    """Test initialization with custom chromium args."""
    custom_args = ["--headless", "--disable-gpu"]
    provider = BrowserInfrastructureProviderLocal(extra_chromium_args=custom_args)
    assert provider.extra_chromium_args == custom_args


def test_browserbase_provider_init_missing_api_key() -> None:
    """Test initialization fails when API key is missing."""
    with (
        patch.dict(os.environ, clear=True),
        pytest.raises(ToolHardError, match="BROWSERBASE_API_KEY is not set"),
    ):
        BrowserInfrastructureProviderBrowserBase()


def test_browserbase_provider_init_missing_project_id() -> None:
    """Test initialization fails when project ID is missing."""
    with (
        patch.dict(os.environ, {"BROWSERBASE_API_KEY": "test_key"}, clear=True),
        pytest.raises(ToolHardError, match="BROWSERBASE_PROJECT_ID is not set"),
    ):
        BrowserInfrastructureProviderBrowserBase()


def test_browserbase_provider_get_context_id_existing(
    mock_browserbase_provider: BrowserInfrastructureProviderBrowserBase,
) -> None:
    """Test getting context ID when already present in end_user additional data."""
    mock_ctx = MagicMock()
    mock_ctx.end_user.get_additional_data.return_value = "existing_context_id"

    context_id = mock_browserbase_provider.get_context_id(mock_ctx, mock_browserbase_provider.bb)

    # Should not create a new context if already present
    mock_browserbase_provider.bb.contexts.create.assert_not_called()  # type: ignore reportFunctionMemberAccess
    assert context_id == "existing_context_id"


def test_browserbase_provider_get_context_id_new(
    mock_browserbase_provider: BrowserInfrastructureProviderBrowserBase,
) -> None:
    """Test getting context ID when not present in end_user additional data."""
    mock_ctx = MagicMock()
    mock_ctx.end_user.get_additional_data.return_value = None
    mock_context = MagicMock()
    mock_context.id = "test_context_id"
    mock_browserbase_provider.bb.contexts.create.return_value = mock_context  # type: ignore reportFunctionMemberAccess

    context_id = mock_browserbase_provider.get_context_id(mock_ctx, mock_browserbase_provider.bb)

    mock_browserbase_provider.bb.contexts.create.assert_called_once_with(project_id="test_project")  # type: ignore reportFunctionMemberAccess
    mock_ctx.end_user.set_additional_data.assert_called_once_with(
        "bb_context_id", "test_context_id"
    )
    assert context_id == "test_context_id"


def test_browserbase_provider_create_session(
    mock_browserbase_provider: BrowserInfrastructureProviderBrowserBase,
) -> None:
    """Test creating a new session."""
    mock_session = MagicMock()
    mock_browserbase_provider.bb.sessions.create.return_value = mock_session  # type: ignore reportFunctionMemberAccess

    session = mock_browserbase_provider.create_session("test_context_id")

    mock_browserbase_provider.bb.sessions.create.assert_called_once_with(  # type: ignore reportFunctionMemberAccess
        project_id="test_project",
        browser_settings={
            "context": {
                "id": "test_context_id",
                "persist": True,
            },
            "solve_captchas": False,
        },
        keep_alive=True,
    )
    assert session == mock_session


def test_browserbase_provider_get_or_create_session_new(
    mock_browserbase_provider: BrowserInfrastructureProviderBrowserBase,
) -> None:
    """Test getting or creating a new session when none exists."""
    context = get_test_tool_context()
    mock_session = MagicMock()
    mock_session.id = "test_session_id"
    mock_session.connect_url = "test_connect_url"

    mock_context = MagicMock()
    mock_context.id = "test_context_id"
    mock_browserbase_provider.bb.contexts.create.return_value = mock_context  # type: ignore reportFunctionMemberAccess
    mock_browserbase_provider.bb.sessions.create.return_value = mock_session  # type: ignore reportFunctionMemberAccess

    connect_url = mock_browserbase_provider.get_or_create_session(
        context, mock_browserbase_provider.bb
    )

    assert connect_url == "test_connect_url"
    assert context.end_user.get_additional_data("bb_session_id") == "test_session_id"
    assert context.end_user.get_additional_data("bb_session_connect_url") == "test_connect_url"
    assert context.end_user.get_additional_data("bb_context_id") == "test_context_id"


def test_browserbase_provider_construct_auth_clarification_url(
    mock_browserbase_provider: BrowserInfrastructureProviderBrowserBase,
) -> None:
    """Test constructing auth clarification URL."""
    context = get_test_tool_context()
    context.end_user.set_additional_data("bb_session_id", "test_session_id")

    mock_debug = MagicMock()
    mock_debug.pages = [MagicMock(debugger_fullscreen_url="https://debug.example.com")]
    mock_browserbase_provider.bb.sessions.debug.return_value = mock_debug  # type: ignore reportFunctionMemberAccess

    url = mock_browserbase_provider.construct_auth_clarification_url(
        context, "https://example.com/login"
    )

    mock_browserbase_provider.bb.sessions.debug.assert_called_once_with("test_session_id")  # type: ignore reportFunctionMemberAccess
    assert str(url) == "https://debug.example.com/"


def test_browserbase_provider_construct_auth_clarification_url_no_session(
    mock_browserbase_provider: BrowserInfrastructureProviderBrowserBase,
) -> None:
    """Test constructing auth clarification URL with no session ID."""
    context = get_test_tool_context()

    with pytest.raises(ToolHardError, match="Session ID not found"):
        mock_browserbase_provider.construct_auth_clarification_url(
            context, "https://example.com/login"
        )


def test_browserbase_provider_setup_browser(
    mock_browserbase_provider: BrowserInfrastructureProviderBrowserBase,
) -> None:
    """Test setting up browser."""
    context = get_test_tool_context()

    mock_session = MagicMock()
    mock_session.id = "test_session_id"
    mock_session.connect_url = "test_connect_url"

    mock_context = MagicMock()
    mock_context.id = "test_context_id"
    mock_browserbase_provider.bb.contexts.create.return_value = mock_context  # type: ignore reportFunctionMemberAccess
    mock_browserbase_provider.bb.sessions.create.return_value = mock_session  # type: ignore reportFunctionMemberAccess

    browser = mock_browserbase_provider.setup_browser(context)

    assert isinstance(browser, Browser)
    assert browser.config.cdp_url == "test_connect_url"


def test_browser_tool_for_url_init_default_parameters() -> None:
    """Test BrowserToolForUrl initialization with default parameters."""
    tool = BrowserToolForUrl()

    assert tool.id == "browser_tool"
    assert tool.name == "Browser Tool"
    assert tool.description == (
        "General purpose browser tool. Can be used to navigate to a URL and complete tasks. "
        "Should only be used if the task requires a browser and you are sure of the URL. "
        "This tool handles a full end to end task. It is capable of doing multiple things "
        "across different URLs within the same root domain as part of the end to end task. As "
        "a result, do not call this tool more than once back to back unless it is for "
        "different root domains - just call it once with the combined task and the URL set "
        "to the root domain."
    )
    assert tool.args_schema == BrowserToolForUrlSchema


def test_browser_tool_for_url_init_custom_parameters() -> None:
    """Test BrowserToolForUrl initialization with custom parameters."""
    custom_id = "custom_browser_tool"
    custom_name = "Custom Browser Tool"
    custom_description = "Custom description for browser tool"
    allowed_domains = ["example.com", "test.com"]

    tool = BrowserToolForUrl(
        id=custom_id,
        name=custom_name,
        description=custom_description,
        allowed_domains=allowed_domains,
    )

    assert tool.id == custom_id
    assert tool.name == custom_name
    assert tool.description == custom_description
    assert tool.args_schema == BrowserToolForUrlSchema
    assert tool.allowed_domains == allowed_domains


def test_browser_tool_for_url_init_subdomain_handling() -> None:
    """Test BrowserToolForUrl initialization correctly handles subdomains."""
    allowed_domains = ["sub.example.com"]
    tool = BrowserToolForUrl(allowed_domains=allowed_domains)

    assert tool.id == "browser_tool"
    assert tool.name == "Browser Tool"
    assert tool.allowed_domains == allowed_domains


def test_browser_tool_for_url_allowed_domains_validation() -> None:
    """Test BrowserToolForUrl allowed domains validation."""
    allowed_domains = ["example.com", "test.com"]
    tool = BrowserToolForUrl(allowed_domains=allowed_domains)
    context = get_test_tool_context()

    # Test valid domain
    result = tool.run(context, "https://example.com", "test task")
    assert result is not None

    # Test valid subdomain
    result = tool.run(context, "https://sub.example.com", "test task")
    assert result is not None

    # Test invalid domain
    with pytest.raises(ValueError, match="URL domain 'invalid.com' is not in allowed domains"):
        tool.run(context, "https://invalid.com", "test task")


def test_browser_tool_for_url_allowed_domains_empty() -> None:
    """Test BrowserToolForUrl with empty allowed domains allows all domains."""
    tool = BrowserToolForUrl(allowed_domains=[])
    context = get_test_tool_context()

    # Should allow any domain when allowed_domains is empty
    result = tool.run(context, "https://any-domain.com", "test task")
    assert result is not None


def test_browser_tool_for_url_no_allowed_domains() -> None:
    """Test BrowserToolForUrl with no allowed domains allows all domains."""
    tool = BrowserToolForUrl()
    context = get_test_tool_context()

    # Should allow any domain when allowed_domains is None
    result = tool.run(context, "https://any-domain.com", "test task")
    assert result is not None


def test_browser_tool_for_url_invalid_domain_format() -> None:
    """Test BrowserToolForUrl raises error for invalid allowed domains."""
    with pytest.raises(ValueError, match="Invalid domain in allowed_domains"):
        BrowserToolForUrl(allowed_domains=["", "valid.com"])

    with pytest.raises(ValueError, match="Invalid domain in allowed_domains"):
        BrowserToolForUrl(allowed_domains=[123, "valid.com"])  # type: ignore


def test_browser_tool_for_url_run_method_signature() -> None:
    """Test BrowserToolForUrl run method accepts URL parameter."""
    tool = BrowserToolForUrl()
    context = get_test_tool_context()
    
    # Test that run method accepts URL parameter
    with patch.object(tool, '_BrowserTool__run') as mock_run:
        mock_run.return_value = "test result"
        result = tool.run(context, "https://example.com", "test task")
        mock_run.assert_called_once_with(context, "https://example.com", "test task", None)
        assert result == "test result"


class TestStructuredOutputSchema(BaseModel):
    """Test schema for structured output testing."""

    result: str = Field(description="Test result field")
    status: str = Field(description="Test status field")


def test_browser_tool_with_structured_output_schema(
    mock_browser_infrastructure_provider: BrowserInfrastructureProvider,
) -> None:
    """Test the browser tool with structured output schema."""
    # Mock task response data with structured output
    mock_task_response = BrowserTaskOutput[TestStructuredOutputSchema](
        task_output=TestStructuredOutputSchema(
            result="Task completed successfully", status="success"
        ),
        human_login_required=False,
    )

    mock_task_result = MagicMock()
    mock_task_result.final_result.return_value = json.dumps(mock_task_response.model_dump())

    # Create async mock for agent.run()
    mock_run = AsyncMock(return_value=mock_task_result)

    # Patch the Agent class
    with patch("portia.open_source_tools.browser_tool.Agent") as mock_agent:
        # Configure the mock Agent instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = mock_run
        mock_agent.return_value = mock_agent_instance

        browser_tool = BrowserTool(
            custom_infrastructure_provider=mock_browser_infrastructure_provider,
            structured_output_schema=TestStructuredOutputSchema,
        )
        context = get_test_tool_context()

        # Run the tool
        result = browser_tool.run(context, "https://example.com", "test task")

        # Verify Agent was called once
        assert mock_agent.call_count == 1
        assert mock_run.call_count == 1

        # Verify the result is the structured output
        assert isinstance(result, TestStructuredOutputSchema)
        assert result.result == "Task completed successfully"
        assert result.status == "success"


def test_browser_tool_with_structured_output_schema_auth_required(
    mock_browser_infrastructure_provider: BrowserInfrastructureProvider,
) -> None:
    """Test the browser tool with structured output schema when auth is required."""
    # Mock auth response data
    mock_auth_response = BrowserTaskOutput[TestStructuredOutputSchema](
        human_login_required=True,
        login_url="https://example.com/login",
        user_login_guidance="Login to example.com",
    )

    # Create a mock result object
    mock_result = MagicMock()
    mock_result.final_result.return_value = json.dumps(mock_auth_response.model_dump())

    # Create async mock for agent.run()
    mock_run = AsyncMock(return_value=mock_result)

    # Patch the Agent class
    with patch("portia.open_source_tools.browser_tool.Agent") as mock_agent:
        # Configure the mock Agent instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = mock_run
        mock_agent.return_value = mock_agent_instance

        browser_tool = BrowserTool(
            custom_infrastructure_provider=mock_browser_infrastructure_provider,
            structured_output_schema=TestStructuredOutputSchema,
        )
        context = get_test_tool_context()

        # Run the tool
        result = browser_tool.run(context, "https://example.com", "test task")

        # Verify Agent was called with correct parameters
        mock_agent.assert_called_once()
        mock_run.assert_called_once()

        # Verify result is an ActionClarification
        assert isinstance(result, ActionClarification)
        assert_clarification_equality_without_uuid(
            result,
            ActionClarification(
                user_guidance="Login to example.com",
                action_url=HttpUrl("https://example.com/login"),
                plan_run_id=context.plan_run.id,
                require_confirmation=True,
                source="Browser tool",
            ),
        )


def test_browserbase_provider_step_complete_with_session(
    mock_browserbase_provider: BrowserInfrastructureProviderBrowserBase,
) -> None:
    """Test step_complete calls sessions.update when session_id is present."""
    mock_ctx = MagicMock()
    end_user = EndUser(external_id="123", additional_data={"bb_session_id": "session123"})
    mock_ctx.end_user = end_user

    mock_browserbase_provider.step_complete(mock_ctx)

    mock_browserbase_provider.bb.sessions.update.assert_called_once_with(  # type: ignore reportFunctionMemberAccess
        "session123",
        project_id="test_project",
        status="REQUEST_RELEASE",
    )


def test_browserbase_provider_step_complete_without_session(
    mock_browserbase_provider: BrowserInfrastructureProviderBrowserBase,
) -> None:
    """Test step_complete does nothing when session_id is missing."""
    mock_ctx = MagicMock()
    end_user = EndUser(external_id="123", additional_data={})
    mock_ctx.end_user = end_user

    mock_browserbase_provider.step_complete(mock_ctx)

    mock_browserbase_provider.bb.sessions.update.assert_not_called()  # type: ignore reportFunctionMemberAccess


def test_browserbase_provider_get_or_create_session_with_clarifications(
    mock_browserbase_provider: BrowserInfrastructureProviderBrowserBase,
) -> None:
    """Test getting or creating a session when clarifications exist."""
    context = get_test_tool_context()
    context.clarifications = [
        ActionClarification(
            user_guidance="Test guidance",
            plan_run_id=PlanRunUUID(),
            action_url=HttpUrl("https://example.com"),
            source="Browser tool",
            step=0,
        )
    ]
    context.plan_run.outputs.clarifications = context.clarifications
    context.end_user.additional_data = {
        "bb_session_id": "existing_session_id",
        "bb_session_connect_url": "existing_connect_url",
    }

    connect_url = mock_browserbase_provider.get_or_create_session(
        context, mock_browserbase_provider.bb
    )

    assert connect_url == "existing_connect_url"
    mock_browserbase_provider.bb.sessions.create.assert_not_called()  # type: ignore reportFunctionMemberAccess


def test_browserbase_provider_get_or_create_session_without_clarifications(
    mock_browserbase_provider: BrowserInfrastructureProviderBrowserBase,
) -> None:
    """Test getting or creating a session when no clarifications exist."""
    context = get_test_tool_context()
    context.clarifications = []

    mock_session = MagicMock()
    mock_session.id = "test_session_id"
    mock_session.connect_url = "test_connect_url"

    mock_context = MagicMock()
    mock_context.id = "test_context_id"
    mock_browserbase_provider.bb.contexts.create.return_value = mock_context  # type: ignore reportFunctionMemberAccess
    mock_browserbase_provider.bb.sessions.create.return_value = mock_session  # type: ignore reportFunctionMemberAccess

    connect_url = mock_browserbase_provider.get_or_create_session(
        context, mock_browserbase_provider.bb
    )

    assert connect_url == "test_connect_url"
    mock_browserbase_provider.bb.sessions.create.assert_called_once()  # type: ignore reportFunctionMemberAccess


def test_process_task_data() -> None:
    """Check strings are passed through."""
    task_data = "this is the data"
    assert BrowserTool.process_task_data(task_data) == task_data

    task_data = ["this is the data"]
    assert BrowserTool.process_task_data(task_data) == "\n".join(task_data)


def test_browser_tool_multiple_calls(
    mock_browserbase_provider: BrowserInfrastructureProviderBrowserBase,
) -> None:
    """Test step_complete only cleans up on final browser tool call."""
    plan = (
        PlanBuilder()
        .step(task="1st browser tool task", tool_id="browser_tool")
        .step(task="2nd browser tool task", tool_id="browser_tool")
        .step(task="3rd browser tool task", tool_id="browser_tool")
        .build()
    )
    end_user = EndUser(
        external_id="123",
        additional_data={"bb_session_id": "session123", "bb_session_connect_url": "connect_url"},
    )
    mock_ctx = MagicMock()
    mock_ctx.end_user = end_user
    mock_ctx.plan = plan
    mock_ctx.plan_run = PlanRun(plan_id=plan.id, current_step_index=0, end_user_id="test")

    # Test first browser tool call (should set up session and not clean up)
    mock_browserbase_provider.setup_browser(mock_ctx)
    mock_browserbase_provider.bb.sessions.create.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue,reportFunctionMemberAccess]
    mock_browserbase_provider.bb.sessions.create.reset_mock()  # pyright: ignore[reportAttributeAccessIssue, reportFunctionMemberAccess]
    mock_browserbase_provider.step_complete(mock_ctx)
    mock_browserbase_provider.bb.sessions.update.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue, reportFunctionMemberAccess]

    # Test middle browser tool call (should not set up or clean up)
    end_user.set_additional_data("bb_session_id", "session123")
    end_user.set_additional_data("bb_session_connect_url", "connect_url")
    mock_ctx.plan_run.current_step_index = 1
    mock_browserbase_provider.setup_browser(mock_ctx)
    mock_browserbase_provider.bb.sessions.create.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue, reportFunctionMemberAccess]
    mock_browserbase_provider.step_complete(mock_ctx)
    mock_browserbase_provider.bb.sessions.update.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue, reportFunctionMemberAccess]

    # Test final browser tool call (should not set up but should clean up)
    mock_ctx.plan_run.current_step_index = 2
    mock_browserbase_provider.setup_browser(mock_ctx)
    mock_browserbase_provider.bb.sessions.create.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue, reportFunctionMemberAccess]
    mock_browserbase_provider.step_complete(mock_ctx)
    mock_browserbase_provider.bb.sessions.update.assert_called_once_with(  # pyright: ignore[reportAttributeAccessIssue, reportFunctionMemberAccess]
        "session123",
        project_id="test_project",
        status="REQUEST_RELEASE",
    )


class TestBrowserToolAllowedDomains:
    """Test class for browser tool allowed domains functionality."""

    def test_browser_tool_for_url_executes_with_domain_restrictions(self) -> None:
        """Test BrowserToolForUrl with domain restrictions executes successfully."""
        allowed_domains = ["example.com"]
        tool = BrowserToolForUrl(url="https://example.com", allowed_domains=allowed_domains)
        context = get_test_tool_context()

        # Mock successful task execution
        mock_task_response = BrowserTaskOutput(
            task_output="Task completed successfully",
            human_login_required=False,
        )

        mock_task_result = MagicMock()
        mock_task_result.final_result.return_value = json.dumps(mock_task_response.model_dump())
        mock_run = AsyncMock(return_value=mock_task_result)

        with patch("portia.open_source_tools.browser_tool.Agent") as mock_agent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run = mock_run
            mock_agent.return_value = mock_agent_instance

            # This should not raise an error
            result = tool.run(context, "test task")
            assert result == "Task completed successfully"

    def test_browser_tool_with_allowed_domains_success(self) -> None:
        """Test BrowserTool with allowed domains executes successfully."""
        allowed_domains = ["example.com"]
        tool = BrowserTool(allowed_domains=allowed_domains)
        context = get_test_tool_context()

        mock_task_response = BrowserTaskOutput(
            task_output="Task completed successfully", 
            human_login_required=False,
        )

        mock_task_result = MagicMock()
        mock_task_result.final_result.return_value = json.dumps(mock_task_response.model_dump())
        mock_run = AsyncMock(return_value=mock_task_result)

        with patch("portia.open_source_tools.browser_tool.Agent") as mock_agent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run = mock_run
            mock_agent.return_value = mock_agent_instance

            result = tool.run(context, "https://example.com", "test task")
            assert result == "Task completed successfully"

    def test_browser_tool_for_url_domain_validation_fails(self) -> None:
        """Test BrowserToolForUrl fails with invalid domain."""
        allowed_domains = ["example.com"]
        tool = BrowserToolForUrl(url="https://invalid.com", allowed_domains=allowed_domains)
        context = get_test_tool_context()

        with pytest.raises(ValueError, match="URL domain 'invalid.com' is not in allowed domains"):
            tool.run(context, "test task")

    def test_browser_tool_allowed_domains_subdomain_validation(self) -> None:
        """Test BrowserTool allows subdomains of allowed domains."""
        allowed_domains = ["example.com"]
        tool = BrowserTool(allowed_domains=allowed_domains)
        context = get_test_tool_context()

        mock_task_response = BrowserTaskOutput(
            task_output="Task completed successfully",
            human_login_required=False,
        )

        mock_task_result = MagicMock()
        mock_task_result.final_result.return_value = json.dumps(mock_task_response.model_dump())
        mock_run = AsyncMock(return_value=mock_task_result)

        with patch("portia.open_source_tools.browser_tool.Agent") as mock_agent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run = mock_run
            mock_agent.return_value = mock_agent_instance

            # Should allow subdomain
            result = tool.run(context, "https://sub.example.com", "test task")
            assert result == "Task completed successfully"

    def test_browser_tool_allowed_domains_initialization_validation(self) -> None:
        """Test BrowserTool validates allowed_domains during initialization."""
        # Should not raise error for valid domains
        tool = BrowserTool(allowed_domains=["example.com", "test.org"])
        assert tool.allowed_domains == ["example.com", "test.org"]

        # Should raise error for invalid domains
        with pytest.raises(ValueError, match="Invalid domain in allowed_domains"):
            BrowserTool(allowed_domains=["", "valid.com"])

        with pytest.raises(ValueError, match="Invalid domain in allowed_domains"):
            BrowserTool(allowed_domains=[123, "valid.com"])  # type: ignore
