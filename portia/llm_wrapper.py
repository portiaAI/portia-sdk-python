"""Wrapper around different LLM providers allowing us to treat them the same."""

import logging
from enum import Enum
from typing import (
    TypeVar,
)

import instructor
from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from mistralai import Mistral
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from portia.config import Config

logger = logging.getLogger(__name__)

# BoundType for BaseModel
T = TypeVar("T", bound=BaseModel)


class LLMProvider(Enum):
    """Enum of LLM providers."""

    OpenAI = "openai"
    Anthropic = "anthropic"
    MistralAI = "mistralai"


DEFAULT_LLM_PROVIDER = LLMProvider.OpenAI
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_MODEL_TEMPERATURE = 0
DEFAULT_MODEL_SEED = 443


class InvalidProviderError(Exception):
    """Raised when a provider is invalid."""

    def __init__(self, provider: str) -> None:
        """Set custom error message."""
        super().__init__(f"{provider} is not a supported provider")


class LLMWrapper:
    """LLMWrapper class for different LLMs."""

    def __init__(
        self,
        config: Config,
    ) -> None:
        """Initialize the wrapper."""
        self.config = config
        self.llm_provider = config.llm_provider or DEFAULT_LLM_PROVIDER
        self.model_name = config.llm_model_name or DEFAULT_LLM_MODEL
        self.model_temperature = config.llm_model_temperature or DEFAULT_MODEL_TEMPERATURE
        self.model_seed = config.llm_model_seed or DEFAULT_MODEL_SEED

    def to_langchain(self) -> BaseChatModel:
        """Return a langchain chat model."""
        match self.llm_provider:
            case LLMProvider.OpenAI:
                return ChatOpenAI(
                    name=self.model_name,
                    temperature=self.model_temperature,
                    seed=self.model_seed,
                    api_key=self.config.openai_api_key,
                )
            case LLMProvider.Anthropic:
                return ChatAnthropic(
                    model_name=self.model_name,
                    temperature=self.model_temperature,
                    timeout=10,
                    stop=None,
                    api_key=self.config.must_get_api_key("anthropic_api_key"),
                )
            case LLMProvider.MistralAI:
                return ChatMistralAI(
                    model_name=self.model_name,
                    temperature=self.model_temperature,
                    api_key=self.config.mistralai_api_key,
                )
            case _:
                raise InvalidProviderError(self.llm_provider)

    def to_instructor(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
    ) -> T:
        """Use instructor to generate an object of response_model type."""
        match self.llm_provider:
            case LLMProvider.OpenAI:
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
            case LLMProvider.Anthropic:
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
                    seed=self.model_seed,
                )
            case LLMProvider.MistralAI:
                client = instructor.from_mistral(
                    client=Mistral(
                        api_key=self.config.must_get_raw_api_key("mistralai_api_key"),
                    ),
                )
                return client.chat.completions.create(
                    model=self.model_name,
                    response_model=response_model,
                    messages=messages,
                    temperature=self.model_temperature,
                    seed=self.model_seed,
                )
            case _:
                raise InvalidProviderError(self.llm_provider)
