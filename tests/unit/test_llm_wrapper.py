"""Test LLM Wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import patch

import pytest
from pydantic import BaseModel, SecretStr

from portia.config import EXECUTION_MODEL_KEY, Config, LLMModel, LLMProvider
from portia.llm_wrapper import BaseLLMWrapper, LLMWrapper, T
from portia.planning_agents.base_planning_agent import StepsOrError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.language_models.chat_models import BaseChatModel
    from openai.types.chat import ChatCompletionMessageParam


def test_base_classes() -> None:
    """Test PlanStorage raises."""

    class MyWrapper(BaseLLMWrapper):
        """Override to test base."""

        def to_instructor(
            self,
            response_model: type[T],
            messages: list[ChatCompletionMessageParam],
        ) -> T:
            return super().to_instructor(response_model, messages)  # type: ignore  # noqa: PGH003

        def to_langchain(self) -> BaseChatModel:
            return super().to_langchain()  # type: ignore  # noqa: PGH003

    wrapper = MyWrapper(SecretStr("test123"))

    with pytest.raises(NotImplementedError):
        wrapper.to_instructor(
            response_model=StepsOrError,
            messages=[],
        )

    with pytest.raises(NotImplementedError):
        wrapper.to_langchain()


@pytest.fixture
def mock_import_check(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Mock the import check."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test123")
    monkeypatch.setenv("GOOGLE_API_KEY", "test123")
    with patch("importlib.util.find_spec", return_value=None):
        yield


class DummyModel(BaseModel):
    """Dummy model for testing."""

    name: str


@pytest.mark.usefixtures("mock_import_check")
@pytest.mark.parametrize("provider", [
    LLMProvider.MISTRALAI,
    LLMProvider.GOOGLE_GENERATIVE_AI,
])
def test_error_if_extension_not_installed_to_langchain(
    provider: LLMProvider,
) -> None:
    """Test that an error is raised if the extension is not installed."""
    llm_wrapper = LLMWrapper.for_usage(
        EXECUTION_MODEL_KEY,
        Config.from_default(llm_provider=provider),
    )

    with pytest.raises(ImportError):
        llm_wrapper.to_langchain()

    with pytest.raises(ImportError):
        llm_wrapper.to_instructor(response_model=DummyModel, messages=[])


def test_construct_azure_openai_llm_wrapper() -> None:
    """Test construct azure openai llm wrapper.

    This test wouldn't be strictly necessary if we had e2e tests for Azure OpenAI
    but we don't (MS portal access problems).
    """
    llm_wrapper = LLMWrapper.for_usage(
        EXECUTION_MODEL_KEY,
        Config.from_default(
            llm_provider=LLMProvider.AZURE_OPENAI,
            azure_openai_endpoint="https://test-azure-openai-endpoint",
            azure_openai_api_key="test-azure-openai-api-key",
        ),
    )
    assert llm_wrapper is not None
    assert llm_wrapper.model_name.provider() == LLMProvider.AZURE_OPENAI
    assert llm_wrapper.api_key == SecretStr("test-azure-openai-api-key")
    assert llm_wrapper.api_endpoint == "https://test-azure-openai-endpoint"

    assert llm_wrapper.to_langchain() is not None
    with mock.patch("instructor.patch", autospec=True):
        assert llm_wrapper.to_instructor(
            response_model=DummyModel,
            messages=[{"role": "system", "content": "test"}],
        ) is not None

    llm_wrapper_no_endpoint = LLMWrapper(
        model_name=LLMModel.AZURE_GPT_4_O,
        api_key=SecretStr("test-azure-openai-api-key"),
        api_endpoint=None,
    )
    with (
        mock.patch("instructor.patch", autospec=True),
        pytest.raises(ValueError, match="endpoint is required"),
    ):
        llm_wrapper_no_endpoint.to_instructor(response_model=DummyModel, messages=[])
