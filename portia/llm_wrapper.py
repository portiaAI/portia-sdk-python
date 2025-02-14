"""Wrapper around different LLM providers, standardizing their usage.

This module provides an abstraction layer around various large language model (LLM) providers,
allowing them to be treated uniformly in the application. It defines a base class `BaseLLMWrapper`
and a concrete implementation `LLMWrapper` that handles communication with different LLM providers
such as OpenAI, Anthropic, and MistralAI.

The `LLMWrapper` class includes methods to convert the provider's model to LangChain-compatible
models and to generate responses using the instructor tool.

Classes in this file include:

- `BaseLLMWrapper`: An abstract base class for all LLM wrappers, providing a template for conversion
methods.
- `LLMWrapper`: A concrete implementation that supports different LLM providers and provides
functionality for converting to LangChain models and generating responses using instructor.

"""

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
    """Abstract base class for LLM wrappers.

    This abstract class defines the interface that all LLM wrappers should implement.
    It requires conversion methods for LangChain models (`to_langchain`) and for generating
    responses using the instructor tool (`to_instructor`).

    Methods:
        to_langchain: Convert the LLM to a LangChain-compatible model.
        to_instructor: Generate a response using the instructor tool.

    """

    def __init__(self, config: Config) -> None:
        """Initialize the base LLM wrapper.

        Args:
            config (Config): The configuration object containing settings for the LLM.

        """
        self.config = config

    @abstractmethod
    def to_langchain(self) -> BaseChatModel:
        """Return a LangChain chat model based on the LLM provider.

        Converts the LLM provider's model to a LangChain-compatible model for interaction
        within the LangChain framework.

        Returns:
            BaseChatModel: A LangChain-compatible model.

        Raises:
            NotImplementedError: If the function is not implemented

        """
        raise NotImplementedError("to_langchain is not implemented")

    @abstractmethod
    def to_instructor(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
    ) -> T:
        """Generate a response using instructor.

        Args:
            response_model (type[T]): The Pydantic model to deserialize the response into.
            messages (list[ChatCompletionMessageParam]): The messages to send to the LLM.

        Returns:
            T: The deserialized response.

        Raises:
            NotImplementedError: If the function is not implemented

        """
        raise NotImplementedError("to_instructor is not implemented")


class LLMWrapper(BaseLLMWrapper):
    """LLMWrapper class for different LLMs.

    This class provides functionality for working with various LLM providers, such as OpenAI,
    Anthropic, and MistralAI. It includes methods to convert the LLM provider's model to a
    LangChain-compatible model and to generate responses using the instructor tool.

    Attributes:
        llm_provider (LLMProvider): The LLM provider to use (e.g., OpenAI, Anthropic, MistralAI).
        model_name (str): The name of the model to use.
        model_temperature (float): The temperature setting for the model.
        model_seed (int): The seed for the model's random generation.

    Methods:
        to_langchain: Converts the LLM provider's model to a LangChain-compatible model.
        to_instructor: Generates a response using instructor for the selected LLM provider.

    """

    def __init__(
        self,
        config: Config,
    ) -> None:
        """Initialize the wrapper.

        Args:
            config (Config): The configuration object containing settings for the LLM.

        """
        super().__init__(config)
        self.llm_provider = config.llm_provider
        self.model_name = config.llm_model_name.value
        self.model_temperature = config.llm_model_temperature
        self.model_seed = config.llm_model_seed

    def to_langchain(self) -> BaseChatModel:
        """Return a LangChain chat model based on the LLM provider.

        Converts the LLM provider's model to a LangChain-compatible model for interaction
        within the LangChain framework.

        Returns:
            BaseChatModel: A LangChain-compatible model.

        """
        match self.llm_provider:
            case LLMProvider.OPENAI:
                return ChatOpenAI(
                    name=self.model_name,
                    model=self.model_name,
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
        """Use instructor to generate an object of the specified response model type.

        Args:
            response_model (type[T]): The Pydantic model to deserialize the response into.
            messages (list[ChatCompletionMessageParam]): The messages to send to the LLM.

        Returns:
            T: The deserialized response from the LLM provider.

        """
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
