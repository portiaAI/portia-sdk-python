"""Configuration for integration tests with comprehensive mocking and API key handling."""

import asyncio
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from portia import Portia
from portia.config import Config, StorageClass
from portia.end_user import EndUser
from portia.tool import Tool, ToolRunContext


class MockWeatherSchema(BaseModel):
    """Schema for mock weather tool."""

    location: str = Field(description="Location to get weather for")


class MockWeatherTool(Tool[str]):
    """Mock weather tool that returns predictable results."""

    id: str = "weather_tool"
    name: str = "Weather Tool"
    description: str = "Get current weather information for a location"
    schema: type[BaseModel] = MockWeatherSchema
    should_summarize: bool = True

    def run(self, context: ToolRunContext, location: str) -> str:  # noqa: ARG002
        """Mock weather data."""
        return f"The weather in {location} is sunny with a temperature of 72°F (22°C). Clear skies."

    async def arun(self, context: ToolRunContext, location: str) -> str:  # noqa: ARG002
        """Async mock weather data."""
        await asyncio.sleep(0.1)  # Simulate network delay
        return f"The weather in {location} is sunny with a temperature of 72°F (22°C). Clear skies."


class MockGmailSearchSchema(BaseModel):
    """Schema for mock Gmail search tool."""

    query: str = Field(description="Search query for emails")


class MockGmailSearchTool(Tool[str]):
    """Mock Gmail search tool."""

    id: str = "portia:google:gmail:search_email"
    name: str = "Gmail Search"
    description: str = "Search emails in Gmail"
    schema: type[BaseModel] = MockGmailSearchSchema
    should_summarize: bool = True

    def run(self, context: ToolRunContext, query: str) -> str:  # noqa: ARG002
        """Mock Gmail search results."""
        return f"Found 3 emails matching '{query}': Meeting, Project update, Lunch plans"

    async def arun(self, context: ToolRunContext, query: str) -> str:  # noqa: ARG002
        """Async mock Gmail search results."""
        await asyncio.sleep(0.1)
        return f"Found 3 emails matching '{query}': Meeting, Project update, Lunch plans"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """Set up comprehensive test environment with all required API keys and configurations."""
    # Set dummy API keys for all providers to prevent config validation errors
    api_keys = {
        "OPENAI_API_KEY": "sk-test-dummy-openai-key-for-integration-tests-1234567890abcdef",
        "ANTHROPIC_API_KEY": "sk-ant-dummy-anthropic-key-for-integration-tests-1234567890abcdef",
        "GOOGLE_API_KEY": "dummy-google-api-key-for-integration-tests-1234567890abcdef",
        "MISTRAL_API_KEY": "dummy-mistral-api-key-for-integration-tests-1234567890abcdef",
        "AZURE_OPENAI_API_KEY": "dummy-azure-openai-key-for-integration-tests-1234567890abcdef",
        "AZURE_OPENAI_ENDPOINT": "https://dummy-azure-endpoint.openai.azure.com/",
        "GROQ_API_KEY": "gsk_dummy-groq-api-key-for-integration-tests-1234567890abcdef",
        "META_API_KEY": "dummy-meta-api-key-for-integration-tests-1234567890abcdef",
        "META_BASE_URL": "https://dummy-meta-endpoint.com/v1",
        "OPENROUTER_API_KEY": "sk-or-dummy-openrouter-key-for-integration-tests-1234567890abcdef",
        "TAVILY_API_KEY": "tvly-dummy-tavily-api-key-for-integration-tests-1234567890abcdef",
        "PORTIA_API_KEY": "dummy-portia-api-key-for-integration-tests-1234567890abcdef",
        "WEATHER_API_KEY": "dummy-weather-api-key-for-integration-tests-1234567890abcdef",
        "AWS_ACCESS_KEY_ID": "dummy-aws-access-key-for-integration-tests",
        "AWS_SECRET_ACCESS_KEY": "dummy-aws-secret-key-for-integration-tests",
        "AWS_DEFAULT_REGION": "us-east-1",
    }

    # Set all API keys in environment
    for key, value in api_keys.items():
        os.environ[key] = value


@pytest.fixture
def integration_portia() -> Portia:
    """Create a Portia instance configured for integration testing with mocked dependencies."""
    # Create config with memory storage to avoid cloud dependencies
    config = Config.from_default(
        storage_class=StorageClass.MEMORY,
        portia_api_key=None,  # Disable cloud features
        default_log_level="DEBUG",
    )

    # Create Portia instance
    portia = Portia(config=config)

    # Add mock tools to the registry
    mock_weather_tool = MockWeatherTool()
    mock_gmail_tool = MockGmailSearchTool()

    # Register mock tools
    portia.tool_registry.add_tool(mock_weather_tool)
    portia.tool_registry.add_tool(mock_gmail_tool)

    return portia


@pytest.fixture(scope="session", autouse=True)
def mock_openai_globally() -> Any:  # noqa: ANN401
    """Mock OpenAI API calls globally to prevent 401 errors from dummy keys."""

    class MockOpenAIResponse:
        """Mock OpenAI response object."""

        def __init__(self, content: str = "Mock response from OpenAI") -> None:
            self.content = content
            self.id = "mock-123"
            self.model = "gpt-4o-mini"
            self.object = "chat.completion"

        @property
        def choices(self) -> list[Any]:
            """Mock choices."""

            class MockChoice:
                def __init__(self, content: str) -> None:
                    self.message = type("Message", (), {"content": content})()
                    self.finish_reason = "stop"

            return [MockChoice(self.content)]

    class MockOpenAIClient:
        """Mock OpenAI client."""

        class Chat:
            class Completions:
                def create(self, **kwargs: Any) -> MockOpenAIResponse:  # noqa: ARG002
                    return MockOpenAIResponse("Mock response from integration test")

                async def acreate(self, **kwargs: Any) -> MockOpenAIResponse:  # noqa: ARG002
                    return MockOpenAIResponse("Mock async response from integration test")

            completions = Completions()

        chat = Chat()

    # Mock at the OpenAI client level
    with (
        patch("openai.OpenAI", return_value=MockOpenAIClient()),
        patch("openai.AsyncOpenAI", return_value=MockOpenAIClient()),
        patch("langchain_openai.ChatOpenAI") as mock_chat_openai,
    ):
        # Configure LangChain mock to return simple responses
        mock_instance = MagicMock()
        mock_instance.invoke.return_value.content = "Mock LangChain response"
        mock_instance.ainvoke.return_value.content = "Mock async LangChain response"
        mock_chat_openai.return_value = mock_instance

        yield


@pytest.fixture
def mock_llm_responses() -> Any:
    """Mock LLM responses to prevent actual API calls."""
    mock_responses = {
        "planning": {
            "steps": [
                {
                    "step": "Get weather for the specified location",
                    "tool": "weather_tool",
                    "output": "$weather_info",
                },
                {
                    "step": "Generate a poem based on the weather",
                    "tool": "llm_tool",
                    "output": "$poem",
                },
            ]
        },
        "execution": "Task completed successfully based on the weather information.",
        "structured": {"result": "success", "data": "mock_data"},
    }

    with (
        patch("portia.model.GenerativeModel.get_response") as mock_get_response,
        patch("portia.model.GenerativeModel.aget_response") as mock_aget_response,
        patch("portia.model.GenerativeModel.get_structured_response") as mock_get_structured,
        patch("portia.model.GenerativeModel.aget_structured_response") as mock_aget_structured,
    ):
        mock_get_response.return_value = mock_responses["execution"]
        mock_aget_response.return_value = mock_responses["execution"]
        mock_get_structured.return_value = mock_responses["planning"]
        mock_aget_structured.return_value = mock_responses["planning"]

        yield {
            "get_response": mock_get_response,
            "aget_response": mock_aget_response,
            "get_structured_response": mock_get_structured,
            "aget_structured_response": mock_aget_structured,
        }


@pytest.fixture
def mock_storage_operations() -> Any:
    """Mock storage operations to prevent network calls."""
    with (
        patch("portia.storage.PortiaCloudStorage.save_plan") as mock_save_plan,
        patch("portia.storage.PortiaCloudStorage.save_plan_run") as mock_save_run,
        patch("portia.storage.PortiaCloudStorage.asave_plan") as mock_asave_plan,
        patch("portia.storage.PortiaCloudStorage.asave_plan_run") as mock_asave_run,
        patch("portia.storage.PortiaCloudStorage.asave_tool_call") as mock_asave_tool,
        patch("portia.storage.PortiaCloudStorage.asave_end_user") as mock_asave_user,
    ):
        # Configure mocks to return success
        mock_save_plan.return_value = None
        mock_save_run.return_value = None
        mock_asave_plan.return_value = None
        mock_asave_run.return_value = None
        mock_asave_tool.return_value = None
        mock_asave_user.return_value = EndUser(external_id="test-user")

        yield {
            "save_plan": mock_save_plan,
            "save_plan_run": mock_save_run,
            "asave_plan": mock_asave_plan,
            "asave_plan_run": mock_asave_run,
            "asave_tool_call": mock_asave_tool,
            "asave_end_user": mock_asave_user,
        }


@pytest.fixture
def mock_docker() -> Any:
    """Mock Docker operations for Redis/cache tests."""
    mock_container = MagicMock()
    mock_container.status = "running"
    mock_container.attrs = {"NetworkSettings": {"IPAddress": "127.0.0.1"}}

    mock_client = MagicMock()
    mock_client.containers.run.return_value = mock_container
    mock_client.containers.get.return_value = mock_container

    with patch("docker.from_env", return_value=mock_client):
        yield mock_client


def pytest_configure(config: Any) -> None:
    """Configure pytest markers for integration tests."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "requires_api_key: mark test as requiring real API keys")
    config.addinivalue_line("markers", "requires_docker: mark test as requiring Docker")


def pytest_collection_modifyitems(config: Any, items: Any) -> None:  # noqa: ARG001
    """Modify test collection to handle integration test requirements."""
    # Check if we're in CI or have specific API keys set for real testing
    has_real_openai_key = os.getenv("OPENAI_API_KEY", "").startswith("sk-") and not os.getenv(
        "OPENAI_API_KEY", ""
    ).startswith("sk-test-")
    has_docker = True
    try:
        import docker

        docker.from_env().ping()
    except Exception:
        has_docker = False

    for item in items:
        # Skip tests requiring real API keys if we don't have them
        if "requires_api_key" in item.keywords and not has_real_openai_key:
            item.add_marker(
                pytest.mark.skip(reason="Requires real API key for integration testing")
            )

        # Skip tests requiring Docker if Docker is not available
        if "requires_docker" in item.keywords and not has_docker:
            item.add_marker(pytest.mark.skip(reason="Requires Docker for integration testing"))
