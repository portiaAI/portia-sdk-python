"""Integration tests for Model subclasses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, SecretStr

from portia._unstable.model import (
    AnthropicModel,
    AzureOpenAIModel,
    GoogleGenerativeAIModel,
    Message,
    MistralAIModel,
    Model,
    OpenAIModel,
)
from portia.config import Config
from portia.planning_agents.base_planning_agent import StepsOrError

if TYPE_CHECKING:
    from collections.abc import Iterator


class Response(BaseModel):
    """Test response model."""

    message: str


CONFIG = Config.from_default()
MODELS: list[Model] = [
    OpenAIModel(model_name="gpt-4o-mini", api_key=CONFIG.openai_api_key),
    AnthropicModel(model_name="claude-3-5-sonnet-latest", api_key=CONFIG.anthropic_api_key),
    MistralAIModel(model_name="mistral-small-latest", api_key=CONFIG.mistralai_api_key),
    GoogleGenerativeAIModel(model_name="gemini-2.0-flash", api_key=CONFIG.google_api_key),
    AZURE_MODEL := AzureOpenAIModel(
        model_name="gpt-4o-mini",
        api_key=SecretStr("dummy"),
        azure_endpoint="dummy",
    ),
]


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
            new_kwargs["api_key"] = CONFIG.openai_api_key.get_secret_value()
            super().__init__(*args, **new_kwargs)

    with mock.patch.object(
        AZURE_MODEL,
        "_client",
        ChatOpenAI(model="gpt-4o-mini", api_key=CONFIG.openai_api_key),
    ), mock.patch("portia._unstable.model.AzureOpenAI", new=AzureOpenAIWrapper):
        yield

@pytest.fixture
def messages() -> list[Message]:
    """Create test messages."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Generate me a random output."),
    ]


@pytest.mark.parametrize("model", MODELS)
def test_get_response(model: Model, messages: list[Message]) -> None:
    """Test get_response for each model type."""
    response = model.get_response(messages)
    assert isinstance(response, Message)
    assert response.role is not None
    assert response.content is not None


@pytest.mark.parametrize("model", MODELS)
def test_get_structured_response(model: Model, messages: list[Message]) -> None:
    """Test get_structured_response for each model type."""
    response = model.get_structured_response(messages, Response)
    assert isinstance(response, Response)
    assert response.message is not None


@pytest.mark.parametrize("model", MODELS)
def test_get_structured_response_steps_or_error(model: Model, messages: list[Message]) -> None:
    """Test get_structured_response with StepsOrError for each model type."""
    response = model.get_structured_response(messages, StepsOrError)
    assert isinstance(response, StepsOrError)
