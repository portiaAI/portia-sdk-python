"""Integration tests for Model subclasses."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from unittest import mock

try:
    import ollama  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    ollama = None  # type: ignore[assignment]
import pytest
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel

from portia.config import Config
from portia.model import GoogleGenAiGenerativeModel, LLMProvider, Message
from portia.planning_agents.base_planning_agent import StepsOrError

if TYPE_CHECKING:
    from collections.abc import Iterator


class Response(BaseModel):
    """Test response model."""

    message: str


# Only test OpenAI to avoid API key and dependency issues in integration tests
MODELS = [
    "openai/gpt-4o-mini",
]

# Skip low capability models that require optional dependencies
LOW_CAPABILITY_MODELS: list[str] = []


@pytest.fixture(autouse=True)
def ollama_model() -> None:
    """Ensure Ollama model is available."""
    if ollama is not None:
        ollama.pull("qwen2.5:0.5b")


@pytest.fixture(autouse=True)
def patch_azure_model() -> Iterator[None]:
    """Patch the Azure model to use the OpenAI client under the hood.

    When we have Azure access we can remove this patch.
    """

    class AzureOpenAIWrapper(OpenAI):
        """Mock the AzureOpenAI client."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            new_kwargs = kwargs.copy()
            new_kwargs.pop("api_version")
            new_kwargs.pop("azure_endpoint")
            super().__init__(*args, **new_kwargs)

    class AzureChatOpenAIWrapper(ChatOpenAI):
        """Mock the AzureChatOpenAI client."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            new_kwargs = kwargs.copy()
            new_kwargs.pop("api_version")
            new_kwargs.pop("azure_endpoint")
            super().__init__(*args, **new_kwargs)

    with (
        mock.patch("portia.model.AzureChatOpenAI", AzureChatOpenAIWrapper),
        mock.patch("portia.model.AzureOpenAI", AzureOpenAIWrapper),
    ):
        yield


@pytest.fixture(autouse=True)
def azure_openai_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture to set the azure openai api key as an env var."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "dummy"))
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://dummy.openai.azure.com")


@pytest.fixture
def messages() -> list[Message]:
    """Create test messages."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Generate me a random output."),
    ]


@pytest.mark.parametrize("model_str", MODELS + LOW_CAPABILITY_MODELS)
def test_get_response(model_str: str, messages: list[Message]) -> None:
    """Test get_response for each model type."""
    model = Config.from_default(default_model=model_str).get_default_model()
    response = model.get_response(messages)
    assert isinstance(response, Message)
    assert response.role is not None
    assert response.content is not None


@pytest.mark.parametrize("model_str", MODELS + LOW_CAPABILITY_MODELS)
@pytest.mark.asyncio
async def test_aget_response(model_str: str, messages: list[Message]) -> None:
    """Test aget_response for each model type."""
    model = Config.from_default(default_model=model_str).get_default_model()
    response = await model.aget_response(messages)
    assert isinstance(response, Message)
    assert response.role is not None
    assert response.content is not None


@pytest.mark.parametrize("model_str", MODELS + LOW_CAPABILITY_MODELS)
@pytest.mark.flaky(reruns=4)
def test_get_structured_response(model_str: str, messages: list[Message]) -> None:
    """Test get_structured_response for each model type."""
    model = Config.from_default(default_model=model_str).get_default_model()
    response = model.get_structured_response(messages, Response)
    assert isinstance(response, Response)
    assert response.message is not None


@pytest.mark.parametrize("model_str", MODELS + LOW_CAPABILITY_MODELS)
@pytest.mark.flaky(reruns=4)
@pytest.mark.asyncio
async def test_aget_structured_response(model_str: str, messages: list[Message]) -> None:
    """Test aget_structured_response for each model type."""
    model = Config.from_default(default_model=model_str).get_default_model()
    response = await model.aget_structured_response(messages, Response)
    assert isinstance(response, Response)
    assert response.message is not None


@pytest.mark.parametrize("model_str", MODELS)
def test_get_structured_response_steps_or_error(model_str: str, messages: list[Message]) -> None:
    """Test get_structured_response with StepsOrError for each model type.

    Skip Ollama models because the small models we used for integration testing aren't
    good enough for complex schemas.
    """
    model = Config.from_default(default_model=model_str).get_default_model()
    response = model.get_structured_response(messages, StepsOrError)
    assert isinstance(response, StepsOrError)


@pytest.mark.parametrize("model_str", MODELS)
@pytest.mark.asyncio
async def test_aget_structured_response_steps_or_error(
    model_str: str, messages: list[Message]
) -> None:
    """Test aget_structured_response with StepsOrError for each model type.

    Skip Ollama models because the small models we used for integration testing aren't
    good enough for complex schemas.
    """
    model = Config.from_default(default_model=model_str).get_default_model()
    response = await model.aget_structured_response(messages, StepsOrError)
    assert isinstance(response, StepsOrError)


def test_google_gemini_temperature(messages: list[Message]) -> None:
    """Test that GoogleGenAiGenerativeModel supports setting temperature."""
    config = Config.from_default(llm_provider=LLMProvider.GOOGLE)
    model = GoogleGenAiGenerativeModel(
        model_name="gemini-2.0-flash",
        api_key=config.google_api_key,
        temperature=0.5,
    )
    response = model.get_response(messages)
    assert isinstance(response, Message)
    assert response.content is not None


@pytest.mark.asyncio
async def test_google_gemini_temperature_async(messages: list[Message]) -> None:
    """Test that GoogleGenAiGenerativeModel supports setting temperature in async mode."""
    config = Config.from_default(llm_provider=LLMProvider.GOOGLE)
    model = GoogleGenAiGenerativeModel(
        model_name="gemini-2.0-flash",
        api_key=config.google_api_key,
        temperature=0.5,
    )
    response = await model.aget_response(messages)
    assert isinstance(response, Message)
    assert response.content is not None
