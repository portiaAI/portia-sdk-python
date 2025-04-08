"""Unit tests for the LLMWrapper."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from portia.config import DEFAULT_MODEL_KEY, Config, LLMModel
from portia.llm_wrapper import LLMWrapper
from portia.model import (
    GenerativeModel,
    LangChainGenerativeModel,
)
from tests.utils import MockToolSchema, get_mock_base_chat_model


def test_llm_wrapper() -> None:
    """Test the LLMWrapper."""
    model = LangChainGenerativeModel(
        client=get_mock_base_chat_model(response=MockToolSchema()),
        model_name="test",
    )

    config = Config(
        custom_models={
            DEFAULT_MODEL_KEY: model,
        },
        openai_api_key=SecretStr("123"),
    )
    wrapper = LLMWrapper.for_usage(config=config, usage=DEFAULT_MODEL_KEY)
    wrapper.to_langchain()
    wrapper.to_instructor(MockToolSchema, [])


def test_llm_wrapper_langchain_not_supported() -> None:
    """Test the LLMWrapper."""
    model = MagicMock(spec=GenerativeModel, create=True)
    wrapper = LLMWrapper(model=model)
    with pytest.raises(
        ValueError,
        match="LangChain is not supported for this model type",
    ):
        wrapper.to_langchain()


def test_llm_wrapper_no_model_name() -> None:
    """Test the LLMWrapper."""
    with pytest.raises(
        ValueError,
        match="model_name and api_key must be provided if model is not provided",
    ):
        LLMWrapper(model=None)


@pytest.mark.parametrize(
    "model_name",
    [
        LLMModel.GPT_4_O,
        LLMModel.CLAUDE_3_5_SONNET,
        LLMModel.MISTRAL_LARGE,
        LLMModel.GEMINI_2_0_FLASH,
        LLMModel.AZURE_GPT_4_O,
    ],
)
def test_llm_wrapper_providers(
    model_name: LLMModel,
) -> None:
    """Test LLMWrapper with different providers."""
    wrapper = LLMWrapper(
        model_name=model_name,
        api_key=SecretStr("test-key"),
        api_endpoint="https://test.example.com",
    )
    assert isinstance(wrapper.to_langchain(), BaseChatModel)

    with patch.object(wrapper.model, "get_structured_response", return_value=MockToolSchema()):
        response = wrapper.to_instructor(MockToolSchema, [])
        assert response == MockToolSchema()
