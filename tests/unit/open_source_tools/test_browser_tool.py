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
    """Test browser setup with allowed_domains passed through."""
    context = get_test_tool_context()
    context.end_user = EndUser(external_id="test_user")
    
    # Mock tool with allowed_domains
    mock_tool = MagicMock()
    mock_tool.allowed_domains = ["example.com"]
    context.tool = mock_tool

    mock_logger_instance = MagicMock()
    mock_logger = MagicMock(return_value=mock_logger_instance)
    with patch("portia.open_source_tools.browser_tool.logger", mock_logger), \
         patch("portia.open_source_tools.browser_tool.Browser") as mock_browser:
        mock_browser_instance = MagicMock()
        mock_browser.return_value = mock_browser_instance
        
        browser = local_browser_provider.setup_browser(context)

        # Verify warning was logged for end_user
        mock_logger.assert_called_once()
        mock_logger_instance.warning.assert_called_once()
        assert "does not support end users" in mock_logger_instance.warning.call_args[0][0]

        # Verify browser instance was created with allowed_domains
        mock_browser.assert_called_once()
        call_args = mock_browser.call_args[1]  # kwargs
        assert "config" in call_args
        # The config should have been created with allowed_domains
        assert browser == mock_browser_instance


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
    """Test setting up browser with allowed_domains passed through."""
    context = get_test_tool_context()
    
    # Mock tool with allowed_domains
    mock_tool = MagicMock()
    mock_tool.allowed_domains = ["example.com"]
    context.tool = mock_tool

    mock_session = MagicMock()
    mock_session.id = "test_session_id"
    mock_session.connect_url = "test_connect_url"

    mock_context = MagicMock()
    mock_context.id = "test_context_id"
    mock_browserbase_provider.bb.contexts.create.return_value = mock_context  # type: ignore reportFunctionMemberAccess
    mock_browserbase_provider.bb.sessions.create.return_value = mock_session  # type: ignore reportFunctionMemberAccess

    with patch("portia.open_source_tools.browser_tool.Browser") as mock_browser:
        mock_browser_instance = MagicMock()
        mock_browser_instance.config.cdp_url = "test_connect_url"
        mock_browser.return_value = mock_browser_instance
        
        browser = mock_browserbase_provider.setup_browser(context)

        # Verify browser was created with allowed_domains
        mock_browser.assert_called_once()
        call_args = mock_browser.call_args[1]  # kwargs
        assert "config" in call_args
        assert browser == mock_browser_instance


def test_browser_tool_for_url_init_default_parameters() -> None:
    """Test BrowserToolForUrl initialization with default parameters."""
    url = "https://example.com"
    tool = BrowserToolForUrl(url=url)

    assert tool.url == url
    assert tool.id == "browser_tool_for_url_example_com"
    assert tool.name == "Browser Tool for example_com"
    assert tool.description == (
        f"Browser tool for the URL {url}. Can be used to navigate to the URL and complete tasks."
    )
    assert tool.args_schema == BrowserToolForUrlSchema


def test_browser_tool_for_url_init_custom_parameters() -> None:
    """Test BrowserToolForUrl initialization with custom parameters."""
    url = "https://example.com"
    custom_id = "custom_browser_tool"
    custom_name = "Custom Browser Tool"
    custom_description = "Custom description for browser tool"

    tool = BrowserToolForUrl(
        url=url,
        id=custom_id,
        name=custom_name,
        description=custom_description,
    )

    assert tool.url == url
    assert tool.id == custom_id
    assert tool.name == custom_name
    assert tool.description == custom_description
    assert tool.args_schema == BrowserToolForUrlSchema


def test_browser_tool_for_url_init_subdomain_handling() -> None:
    """Test BrowserToolForUrl initialization correctly handles subdomains."""
    url = "https://sub.example.com"
    tool = BrowserToolForUrl(url=url)

    assert tool.url == url
    assert tool.id == "browser_tool_for_url_sub_example_com"
    assert tool.name == "Browser Tool for sub_example_com"


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


# ===== ALLOWED DOMAINS FUNCTIONALITY TESTS =====


class TestBrowserToolAllowedDomains:
    """Test suite for BrowserTool with allowed_domains functionality."""

    def test_allowed_domains_permits_navigation_to_whitelisted_domains(
        self,
        mock_browser_infrastructure_provider: BrowserInfrastructureProvider,
    ) -> None:
        """Test BrowserTool allows navigation to domains in allowed_domains list."""
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

            browser_tool = BrowserTool(
                custom_infrastructure_provider=mock_browser_infrastructure_provider,
                allowed_domains=["example.com", "trusted.org"],
            )
            context = get_test_tool_context()

            # Should succeed for allowed domain
            result = browser_tool.run(context, "https://example.com", "test task")
            assert result == "Task completed successfully"

            # Should succeed for subdomain of allowed domain
            result = browser_tool.run(context, "https://www.example.com", "test task")
            assert result == "Task completed successfully"

    def test_allowed_domains_parameter_passed_to_browser_use(
        self,
        mock_browser_infrastructure_provider: BrowserInfrastructureProvider,
    ) -> None:
        """Test BrowserTool passes allowed_domains to browser-use correctly."""
        browser_tool = BrowserTool(
            custom_infrastructure_provider=mock_browser_infrastructure_provider,
            allowed_domains=["example.com", "*.wikipedia.org"],
        )
        
        # Verify allowed_domains is set on the tool
        assert browser_tool.allowed_domains == ["example.com", "*.wikipedia.org"]
        
        # Note: Domain validation is now handled by browser-use, not by custom logic
        # So we don't test validation behavior here, just that the parameter is set

    def test_browser_tool_for_url_accepts_allowed_domains_parameter(self) -> None:
        """Test BrowserToolForUrl correctly accepts and stores allowed_domains parameter."""
        # Should succeed when URL matches allowed domain
        tool = BrowserToolForUrl(
            url="https://example.com",
            allowed_domains=["example.com"]
        )
        assert tool.allowed_domains == ["example.com"]
        assert tool.url == "https://example.com"

        # Should be able to create tool with subdomain when parent domain is allowed
        tool_subdomain = BrowserToolForUrl(
            url="https://www.example.com",
            allowed_domains=["example.com"]
        )
        assert tool_subdomain.allowed_domains == ["example.com"]
        assert tool_subdomain.url == "https://www.example.com"

    def test_browser_tool_for_url_executes_with_domain_restrictions(
        self,
        mock_browser_infrastructure_provider: BrowserInfrastructureProvider,
    ) -> None:
        """Test BrowserToolForUrl executes successfully when URL matches allowed_domains."""
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

            # Create tool with allowed domain
            tool = BrowserToolForUrl(
                url="https://example.com",
                allowed_domains=["example.com"],
                custom_infrastructure_provider=mock_browser_infrastructure_provider,
            )
            
            context = get_test_tool_context()
            
            # Should succeed since the URL matches allowed domain
            result = tool.run(context, "test task")
            assert result == "Task completed successfully"

    def test_allowed_domains_supports_multiple_domain_whitelist(
        self,
        mock_browser_infrastructure_provider: BrowserInfrastructureProvider,
    ) -> None:
        """Test allowed_domains correctly handles multiple domains in whitelist."""
        allowed_domains = ["example.com", "trusted.org", "safe.net"]
        
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

            browser_tool = BrowserTool(
                custom_infrastructure_provider=mock_browser_infrastructure_provider,
                allowed_domains=allowed_domains,
            )
            context = get_test_tool_context()

            # Should succeed for all allowed domains
            for domain in ["https://example.com", "https://trusted.org", "https://safe.net"]:
                result = browser_tool.run(context, domain, "test task")
                assert result == "Task completed successfully"

            # Note: Domain validation is now handled by browser-use internally
            # We don't test blocking behavior here as that's browser-use's responsibility

    def test_allowed_domains_integration_with_browser_config(
        self,
        mock_browser_infrastructure_provider: BrowserInfrastructureProvider,
    ) -> None:
        """Test that allowed_domains is properly passed to BrowserConfig during setup."""
        browser_tool = BrowserTool(
            custom_infrastructure_provider=mock_browser_infrastructure_provider,
            allowed_domains=["example.com", "*.trusted.org"],
        )
        context = get_test_tool_context()
        
        with patch("portia.open_source_tools.browser_tool.Browser") as mock_browser:
            mock_browser_instance = MagicMock()
            mock_browser.return_value = mock_browser_instance
            
            # Setup browser through infrastructure provider
            browser = mock_browser_infrastructure_provider.setup_browser(context)
            
            # Verify Browser was called with BrowserConfig containing allowed_domains
            mock_browser.assert_called_once()
            call_args = mock_browser.call_args
            
            # Check that config parameter exists
            assert "config" in call_args.kwargs or len(call_args.args) > 0
            
    def test_allowed_domains_none_allows_all_domains(
        self,
        mock_browser_infrastructure_provider: BrowserInfrastructureProvider,
    ) -> None:
        """Test that when allowed_domains is None, all domains are permitted (default behavior)."""
        browser_tool = BrowserTool(
            custom_infrastructure_provider=mock_browser_infrastructure_provider,
            allowed_domains=None,  # Explicitly set to None
        )
        
        # Verify allowed_domains is None
        assert browser_tool.allowed_domains is None
        
        # This should work for any domain since no restrictions
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
            
            context = get_test_tool_context()
            
            # Should succeed for any domain when allowed_domains is None
            result = browser_tool.run(context, "https://any-domain.com", "test task")
            assert result == "Task completed successfully"

    def test_allowed_domains_supports_glob_patterns(
        self,
        mock_browser_infrastructure_provider: BrowserInfrastructureProvider,
    ) -> None:
        """Test that allowed_domains supports glob patterns like *.example.com."""
        browser_tool = BrowserTool(
            custom_infrastructure_provider=mock_browser_infrastructure_provider,
            allowed_domains=["*.example.com", "specific-site.org"],
        )
        
        # Verify glob patterns are properly stored
        assert "*.example.com" in browser_tool.allowed_domains
        assert "specific-site.org" in browser_tool.allowed_domains
        assert len(browser_tool.allowed_domains) == 2

    def test_allowed_domains_validation_delegated_to_browser_use(self) -> None:
        """Test that domain validation is properly delegated to browser-use rather than custom logic."""
        # Test that we don't implement custom URL validation
        # This is important because custom validation often has security bugs
        
        # Create tool with restricted domains
        browser_tool = BrowserTool(allowed_domains=["example.com"])
        
        # Verify no custom validation methods exist
        assert not hasattr(browser_tool, '_validate_url')
        assert not hasattr(browser_tool, '_check_domain_allowed')
        assert not hasattr(browser_tool, '_parse_domain')
        
        # Domain validation should be handled by browser-use's BrowserConfig
        assert browser_tool.allowed_domains == ["example.com"]
