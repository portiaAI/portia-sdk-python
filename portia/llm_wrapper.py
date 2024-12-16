"""Wrapper around different LLM providers allowing us to treat them the same."""

import logging
from enum import Enum
from typing import (
    Any,
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

    _instance = None  # Singleton instance

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Return singleton instance if it exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__(*args, **kwargs)
        return cls._instance

    def __init__(
        self,
        config: Config,
    ) -> None:
        """Initialize the wrapper."""
        # set defaults
        llm_provider = config.llm_provider or DEFAULT_LLM_PROVIDER
        model_name = config.llm_model_name or DEFAULT_LLM_MODEL
        model_temperature = config.llm_model_temperature or DEFAULT_MODEL_TEMPERATURE
        model_seed = config.llm_model_seed or DEFAULT_MODEL_SEED

        # Prevent reinitialization in singleton pattern
        if hasattr(self, "_initialized") and self._initialized:
            if (
                llm_provider is not None
                and self._instance
                and self._instance.llm_provider != llm_provider
            ):
                error = (
                    f"""Attempting to reinitialize LLM from {self.llm_provider} to {llm_provider}"""
                )
                raise ValueError(error)
            return

        self.config = config
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.model_temperature = model_temperature
        self.model_seed = model_seed
        self._initialized = True

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
        response_model: type[BaseModel],
        messages: list[ChatCompletionMessageParam],
    ) -> BaseModel:
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
