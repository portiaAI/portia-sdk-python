"""wrapper tests."""

import pytest

from portia.config import Config, LLMModel, LLMProvider
from portia.llm_wrapper import LLMWrapper
from portia.plan import Plan

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
@pytest.mark.flaky(reruns=3)  # MistralAI is a little flaky on the to_instructor call
def test_wrapper_methods(llm_provider: LLMProvider, llm_model_name: LLMModel) -> None:
    """Test we can generate wrappers for important providers."""
    c = Config.from_default(
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
    )

    wrapper = LLMWrapper(c)
    # check we don't get errors
    wrapper.to_instructor(
        Plan,
        [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
        ],
    )
    model = wrapper.to_langchain()
    assert model.model_name == llm_model_name.value
