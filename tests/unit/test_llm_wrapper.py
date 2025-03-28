from unittest.mock import MagicMock

import pytest
from portia.config import DEFAULT_MODEL_KEY, Config
from portia.llm_wrapper import LLMWrapper
from portia.model import LangChainModel, Model
from tests.utils import MockToolSchema, get_mock_base_chat_model


def test_llm_wrapper() -> None:
    """Test the LLMWrapper."""
    config = Config(
        models={
            DEFAULT_MODEL_KEY: LangChainModel(
                client=get_mock_base_chat_model(response=MockToolSchema()),
            ),
        },
    )
    wrapper = LLMWrapper.for_usage(config=config, usage=DEFAULT_MODEL_KEY)
    wrapper.to_langchain()
    wrapper.to_instructor(MockToolSchema, [])


def test_llm_wrapper_langchain_not_supported() -> None:
    """Test the LLMWrapper."""
    model = MagicMock(spec=Model, create=True)
    wrapper = LLMWrapper(model)
    with pytest.raises(
        ValueError,
        match="LangChain is not supported for this model type",
    ):
        wrapper.to_langchain()
