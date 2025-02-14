"""Test LLM Wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from portia.llm_wrapper import BaseLLMWrapper
from tests.utils import get_test_config

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


def test_base_classes() -> None:
    """Test PlanStorage raises."""

    class MyWrapper(BaseLLMWrapper):
        """Override to test base."""

        def to_langchain(self) -> BaseChatModel:
            return super().to_langchain()  # type: ignore  # noqa: PGH003

    wrapper = MyWrapper(get_test_config())

    with pytest.raises(NotImplementedError):
        wrapper.to_langchain()
