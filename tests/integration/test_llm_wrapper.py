"""wrapper tests."""

import pytest

from portia.config import LLMModel, LLMProvider, default_config
from portia.errors import InvalidLLMProviderError
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
def test_wrapper_methods(llm_provider: LLMProvider, llm_model_name: LLMModel) -> None:
    """Test we can generate wrappers for important providers."""
    c = default_config()
    c.llm_provider = llm_provider
    c.llm_model_name = llm_model_name
    wrapper = LLMWrapper(c)
    # check we don't get errors
    wrapper.to_instructor(
        Plan,
        [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
        ],
    )
    wrapper.to_langchain()


def test_wrapper_method_invalid() -> None:
    """Test we can generate wrappers for important providers."""
    c = default_config()
    c.llm_provider = "Invalid"  # type: ignore  # noqa: PGH003
    wrapper = LLMWrapper(c)
    with pytest.raises(InvalidLLMProviderError):
        wrapper.to_instructor(
            Plan,
            [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "test"},
            ],
        )
    with pytest.raises(InvalidLLMProviderError):
        wrapper.to_langchain()
