"""wrapper tests."""

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI

from portia.config import LLMModel, default_config
from portia.llm_wrapper import LLMWrapper
from portia.plan import Plan

MODELS = [
    LLMModel.GPT_4_O,
    LLMModel.MISTRAL_LARGE_LATEST,
    LLMModel.CLAUDE_3_5_SONNET,
]


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.flaky(reruns=3)  # MistralAI is a little flaky on the to_instructor call
def test_wrapper_methods(model_name: LLMModel) -> None:
    """Test we can generate wrappers for important providers."""
    c = default_config(
        llm_model_name=model_name,
    )
    wrapper = LLMWrapper(model_name=model_name, api_key=c.get_llm_api_key(model_name))
    # check we don't get errors
    wrapper.to_instructor(
        Plan,
        [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
        ],
    )
    model = wrapper.to_langchain()

    if isinstance(model, (ChatMistralAI, ChatAnthropic)):
        # MistralAI and Anthropic get_name() method doesn't give the model name unfortunately
        assert model.model == model_name.value
    else:
        assert model.get_name() == model_name.value
