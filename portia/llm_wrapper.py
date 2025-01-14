"""Wrapper around different LLM providers allowing us to treat them the same."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

import instructor
from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from mistralai import Mistral
from openai import OpenAI
from pydantic import BaseModel

from portia.config import Config, LLMProvider

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import (
        BaseChatModel,
    )
    from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseLLMWrapper(ABC):
    """Abstract base class for LLM wrappers."""

    def __init__(self, config: Config) -> None:
        """Initialize the base LLM wrapper."""
        self.config = config

    @abstractmethod
    def to_langchain(self) -> BaseChatModel:
        """Convert to a LangChain-compatible model."""

    @abstractmethod
    def to_instructor(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
    ) -> T:
        """Generate a response using instructor."""


class LLMWrapper(BaseLLMWrapper):
    """LLMWrapper class for different LLMs."""

    def __init__(
        self,
        config: Config,
    ) -> None:
        """Initialize the wrapper."""
        super().__init__(config)
        self.llm_provider = config.llm_provider
        self.model_name = config.llm_model_name.value
        self.model_temperature = config.llm_model_temperature
        self.model_seed = config.llm_model_seed

    def to_langchain(self) -> BaseChatModel:
        """Return a langchain chat model."""
        match self.llm_provider:
            case LLMProvider.OPENAI:
                return ChatOpenAI(
                    name=self.model_name,
                    temperature=self.model_temperature,
                    seed=self.model_seed,
                    api_key=self.config.openai_api_key,
                    max_retries=3,
                )
            case LLMProvider.ANTHROPIC:
                return ChatAnthropic(
                    model_name=self.model_name,
                    temperature=self.model_temperature,
                    timeout=120,
                    stop=None,
                    max_retries=3,
                    api_key=self.config.must_get_api_key("anthropic_api_key"),
                )
            case LLMProvider.MISTRALAI:
                return ChatMistralAI(
                    model_name=self.model_name,
                    temperature=self.model_temperature,
                    api_key=self.config.mistralai_api_key,
                    max_retries=3,
                )

    def to_instructor(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
    ) -> T:
        """Use instructor to generate an object of response_model type."""
        match self.llm_provider:
            case LLMProvider.OPENAI:
                client = instructor.from_openai(
                    client=OpenAI(
                        api_key=self.config.must_get_raw_api_key("openai_api_key"),
                    ),
                    mode=instructor.Mode.JSON,
                )
                return client.chat.completions.create(
                    response_model=response_model,
                    messages=messages,
                    model=self.model_name,
                    temperature=self.model_temperature,
                    seed=self.model_seed,
                )
            case LLMProvider.ANTHROPIC:
                client = instructor.from_anthropic(
                    client=Anthropic(
                        api_key=self.config.must_get_raw_api_key("anthropic_api_key"),
                    ),
                    mode=instructor.Mode.ANTHROPIC_JSON,
                )
                return client.chat.completions.create(
                    model=self.model_name,
                    response_model=response_model,
                    messages=messages,
                    max_tokens=2048,
                    temperature=self.model_temperature,
                )
            case LLMProvider.MISTRALAI:
                client = instructor.from_mistral(
                    client=Mistral(
                        api_key=self.config.must_get_raw_api_key("mistralai_api_key"),
                    ),
                )
                return client.chat.completions.create(  # pyright: ignore[reportReturnType]
                    model=self.model_name,
                    response_model=response_model,
                    messages=messages,
                    temperature=self.model_temperature,
                )
