"""Unit tests for the LLMWrapper."""

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from portia.config import DEFAULT_MODEL_KEY, PLANNING_DEFAULT_MODEL_KEY, Config
from portia.llm_wrapper import LLMWrapper
from portia.model import GenerativeModel, LangChainGenerativeModel
from tests.utils import MockToolSchema, get_mock_base_chat_model


def test_llm_wrapper() -> None:
    """Test the LLMWrapper."""
    model = LangChainGenerativeModel(
        client=get_mock_base_chat_model(response=MockToolSchema()),
        model_name="test",
    )

    config = Config(
        models={
            PLANNING_DEFAULT_MODEL_KEY: model,
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
    wrapper = LLMWrapper(model)
    with pytest.raises(
        ValueError,
        match="LangChain is not supported for this model type",
    ):
        wrapper.to_langchain()
