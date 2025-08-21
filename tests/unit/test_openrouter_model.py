"""Tests for OpenRouter model implementation."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from portia.model import Message, OpenRouterGenerativeModel


class StructuredOutputTestModel(MagicMock):
    """Test model for structured output testing."""


@pytest.mark.asyncio
async def test_openrouter_model_aget_response() -> None:
    """Test OpenRouter model aget_response method."""
    with patch("portia.model.ChatOpenAI") as mock_chat_openrouter_cls:
        mock_chat_openrouter = MagicMock()

        async def mock_ainvoke(*_: Any, **__: Any) -> AIMessage:
            return AIMessage(content="OpenRouter response")

        mock_chat_openrouter.ainvoke = mock_ainvoke
        mock_chat_openrouter_cls.return_value = mock_chat_openrouter

        model = OpenRouterGenerativeModel(
            model_name="moonshotai/kimi-k2",
            api_key=SecretStr("test"),
        )

        messages = [Message(role="user", content="Hello")]
        response = await model.aget_response(messages)

        assert isinstance(response, Message)
        assert response.role == "assistant"
        assert response.content == "OpenRouter response"


@pytest.mark.asyncio
async def test_openrouter_model_aget_structured_response() -> None:
    """Test OpenRouter model aget_structured_response method."""
    with patch("portia.model.ChatOpenAI") as mock_chat_openrouter_cls:
        mock_chat_openrouter = MagicMock()

        structured_output = MagicMock()

        async def mock_structured_ainvoke(*_: Any, **__: Any) -> StructuredOutputTestModel:
            return StructuredOutputTestModel(test_field="OpenRouter structured response")

        structured_output.ainvoke = mock_structured_ainvoke
        mock_chat_openrouter.with_structured_output.return_value = structured_output
        mock_chat_openrouter_cls.return_value = mock_chat_openrouter

        model = OpenRouterGenerativeModel(
            model_name="moonshotai/kimi-k2",
            api_key=SecretStr("test"),
        )

        messages = [Message(role="user", content="Hello")]
        result = await model.aget_structured_response(messages, StructuredOutputTestModel)
        assert isinstance(result, StructuredOutputTestModel)
        assert result.test_field == "OpenRouter structured response"
