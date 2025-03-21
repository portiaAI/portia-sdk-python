"""Test LLM Wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pydantic import BaseModel, SecretStr

from portia.config import EXECUTION_MODEL_KEY, Config, LLMProvider
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
    with patch("importlib.util.find_spec", return_value=None):
        yield


class DummyModel(BaseModel):
    """Dummy model for testing."""

    name: str


@pytest.mark.usefixtures("mock_import_check")
@pytest.mark.parametrize("provider", [LLMProvider.MISTRALAI])
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


@pytest.mark.usefixtures("mock_import_check")
@pytest.mark.parametrize("provider", [LLMProvider.MISTRALAI])
def test_error_if_extension_not_installed_to_instructor(
    provider: LLMProvider,
) -> None:
    """Test that an error is raised if the extension is not installed."""
    llm_wrapper = LLMWrapper.for_usage(
        EXECUTION_MODEL_KEY,
        Config.from_default(llm_provider=provider),
    )

    with pytest.raises(ImportError):
        llm_wrapper.to_instructor(response_model=DummyModel, messages=[])
