from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
import pytest
from pydantic import SecretStr
from portia.model import OpenRouterGenerativeModel, Message, LLMProvider


class StructuredOutputTestModel(MagicMock):
    pass


@pytest.mark.asyncio
async def test_openrouter_model_async_methods() -> None:
    """Test OpenRouter model async methods."""
    with patch("portia.model.ChatOpenAI") as mock_chat_openrouter_cls:
        mock_chat_openrouter = MagicMock()
        
        async def mock_ainvoke(*_, **__) -> AIMessage:
            return AIMessage(content="OpenRouter response")
        mock_chat_openrouter.ainvoke = mock_ainvoke
        
        structured_output = MagicMock()
        
        async def mock_structured_ainvoke(*_, **__) -> StructuredOutputTestModel:
            return StructuredOutputTestModel(test_field="OpenRouter structured response")
        structured_output.ainvoke = mock_structured_ainvoke
        mock_chat_openrouter.with_structured_output.return_value = structured_output
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
        
        # Test aget_structured_response
        result = await model.aget_structured_response(messages, StructuredOutputTestModel)
        assert isinstance(result, StructuredOutputTestModel)
        assert result.test_field == "OpenRouter structured response"