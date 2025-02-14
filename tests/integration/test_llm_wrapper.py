"""wrapper tests."""

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI

from portia.config import Config, LLMModel, LLMProvider
from portia.llm_wrapper import LLMWrapper

PROVIDER_MODELS = [
    (
        LLMProvider.OPENAI,
        LLMModel.GPT_4_O_MINI,
    ),
    (
        LLMProvider.MISTRALAI,
        LLMModel.MISTRAL_LARGE_LATEST,
    ),
    (
        LLMProvider.ANTHROPIC,
        LLMModel.CLAUDE_3_OPUS_LATEST,
    ),
]


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
def test_wrapper_methods(llm_provider: LLMProvider, llm_model_name: LLMModel) -> None:
    """Test we can generate wrappers for important providers."""
    c = Config.from_default(
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
    )

    wrapper = LLMWrapper(c)
    model = wrapper.to_langchain()

    if isinstance(model, (ChatMistralAI, ChatAnthropic)):
        # MistralAI and Anthropic get_name() method doesn't give the model name unfortunately
        assert model.model == llm_model_name.value
    else:
        assert model.get_name() == llm_model_name.value
