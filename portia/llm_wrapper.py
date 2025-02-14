"""Wrapper around different LLM providers, standardizing their usage.

This module provides an abstraction layer around various large language model (LLM) providers,
allowing them to be treated uniformly in the application. It defines a base class `BaseLLMWrapper`
and a concrete implementation `LLMWrapper` that handles communication with different LLM providers
such as OpenAI, Anthropic, and MistralAI.

The `LLMWrapper` class includes methods to convert the provider's model to LangChain-compatible
models.

Classes in this file include:

- `BaseLLMWrapper`: An abstract base class for all LLM wrappers, providing a template for conversion
methods.
- `LLMWrapper`: A concrete implementation that supports different LLM providers and provides
functionality for converting to LangChain models.

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from portia.config import Config, LLMProvider

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import (
        BaseChatModel,
    )

logger = logging.getLogger(__name__)


class BaseLLMWrapper(ABC):
    """Abstract base class for LLM wrappers.

    This abstract class defines the interface that all LLM wrappers should implement.
    It requires conversion methods for LangChain models (`to_langchain`).

    Methods:
        to_langchain: Convert the LLM to a LangChain-compatible model.

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


class LLMWrapper(BaseLLMWrapper):
    """LLMWrapper class for different LLMs.

    This class provides functionality for working with various LLM providers, such as OpenAI,
    Anthropic, and MistralAI. It includes methods to convert the LLM provider's model to a
    LangChain-compatible model.

    Attributes:
        llm_provider (LLMProvider): The LLM provider to use (e.g., OpenAI, Anthropic, MistralAI).
        model_name (str): The name of the model to use.
        model_temperature (float): The temperature setting for the model.
        model_seed (int): The seed for the model's random generation.

    Methods:
        to_langchain: Converts the LLM provider's model to a LangChain-compatible model.

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
