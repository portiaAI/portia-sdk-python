"""Unit tests for Groq model provider integration."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from portia.config import Config
from portia.model import GroqGenerativeModel, LLMProvider, Message


def test_groq_provider_enum() -> None:
    """Test that GROQ is available in LLMProvider enum."""
    assert hasattr(LLMProvider, "GROQ")
    assert LLMProvider.GROQ.value == "groq"


def test_groq_model_instantiation() -> None:
    """Test that GroqGenerativeModel can be instantiated with proper parameters."""
    model = GroqGenerativeModel(
        model_name="llama3-8b-8192",
        api_key=SecretStr("test-groq-api-key"),
        temperature=0.0,
    )

    assert model.provider == LLMProvider.GROQ
    assert model.model_name == "llama3-8b-8192"


def test_groq_config_integration() -> None:
    """Test that Config can be created with Groq provider and API key."""
    config = Config(
        llm_provider=LLMProvider.GROQ,
        groq_api_key=SecretStr("test-groq-api-key"),
    )
    # Set models via the nested `models` structure to satisfy type checker
    config.models.default_model = "groq/llama3-8b-8192"
    config.models.planning_model = "groq/llama3-70b-8192"
    config.models.execution_model = "groq/llama3-8b-8192"
    config.models.introspection_model = "groq/llama3-8b-8192"

    assert config.llm_provider == LLMProvider.GROQ
    assert config.groq_api_key.get_secret_value() == "test-groq-api-key"
    assert config.models.default_model == "groq/llama3-8b-8192"


def test_groq_auto_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Groq provider is auto-detected when GROQ_API_KEY is set."""
    from portia.config import llm_provider_default_from_api_keys

    # Set GROQ_API_KEY in environment
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-api-key")

    # Should auto-detect Groq
    detected_provider = llm_provider_default_from_api_keys()
    assert detected_provider == LLMProvider.GROQ


def test_groq_model_parsing() -> None:
    """Test that groq/<model> strings are parsed correctly."""
    config = Config(
        llm_provider=LLMProvider.GROQ,
        groq_api_key=SecretStr("test-groq-api-key"),
    )
    config.models.default_model = "groq/llama3-8b-8192"

    model = config._parse_model_string("groq/llama3-8b-8192")
    assert isinstance(model, GroqGenerativeModel)
    assert model.provider == LLMProvider.GROQ
    assert model.model_name == "llama3-8b-8192"


@pytest.mark.parametrize(
    "model_name",
    [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
    ],
)
def test_groq_model_names(model_name: str) -> None:
    """Test various Groq model names work correctly."""
    model = GroqGenerativeModel(
        model_name=model_name,
        api_key=SecretStr("test-groq-api-key"),
    )

    assert model.model_name == model_name
    assert model.provider == LLMProvider.GROQ


def test_groq_default_models() -> None:
    """Test that Groq has appropriate default models configured."""
    from portia.config import Config

    config = Config(
        llm_provider=LLMProvider.GROQ,
        groq_api_key=SecretStr("test-groq-api-key"),
    )

    # Test default model assignment
    default_model = config.get_agent_default_model("default_model", LLMProvider.GROQ)
    assert default_model == "groq/llama3-8b-8192"

    planning_model = config.get_agent_default_model("planning_model", LLMProvider.GROQ)
    assert planning_model == "groq/llama3-70b-8192"

    introspection_model = config.get_agent_default_model("introspection_model", LLMProvider.GROQ)
    assert introspection_model == "groq/llama3-8b-8192"


@pytest.mark.asyncio
async def test_groq_model_aget_response() -> None:
    """Ensure Groq aget_response returns a proper Message."""
    with patch("portia.model.ChatOpenAI") as mock_chat_cls:
        mock_chat = MagicMock()

        async def mock_ainvoke(*_: Any, **__: Any) -> AIMessage:
            return AIMessage(content="Groq response")

        mock_chat.ainvoke = mock_ainvoke
        mock_chat_cls.return_value = mock_chat

        model = GroqGenerativeModel(
            model_name="llama3-8b-8192",
            api_key=SecretStr("test-groq-api-key"),
        )

        messages = [Message(role="user", content="Hello")]
        response = await model.aget_response(messages)

        assert isinstance(response, Message)
        assert response.role == "assistant"
        assert response.content == "Groq response"


class _StructuredOutputTestModel(MagicMock):
    """Dummy structured output type for testing."""


@pytest.mark.asyncio
async def test_groq_model_aget_structured_response() -> None:
    """Ensure Groq aget_structured_response returns the provided schema instance."""
    with patch("portia.model.ChatOpenAI") as mock_chat_cls:
        mock_chat = MagicMock()

        structured_output = MagicMock()

        async def mock_structured_ainvoke(*_: Any, **__: Any) -> _StructuredOutputTestModel:
            return _StructuredOutputTestModel(test_field="Groq structured response")

        structured_output.ainvoke = mock_structured_ainvoke
        mock_chat.with_structured_output.return_value = structured_output
        mock_chat_cls.return_value = mock_chat

        model = GroqGenerativeModel(
            model_name="llama3-8b-8192",
            api_key=SecretStr("test-groq-api-key"),
        )

        messages = [Message(role="user", content="Hello")]
        result = await model.aget_structured_response(messages, _StructuredOutputTestModel)

        assert isinstance(result, _StructuredOutputTestModel)
        assert getattr(result, "test_field", "") == "Groq structured response"
