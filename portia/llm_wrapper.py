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
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langsmith import wrappers
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, SecretStr

from portia.config import Config, LLMModel, LLMProvider, validate_extras_dependencies

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

    def __init__(self, api_key: SecretStr) -> None:
        """Initialize the base LLM wrapper.

        Args:
            api_key (str): The API key for the LLM provider.

        """
        self.api_key = api_key

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
        model_name (LLMModel): The name of the model to use.
        api_key (SecretStr): The API key for the LLM provider.
        model_seed (int): The seed for the model's random generation.
        api_endpoint (str | None): The API endpoint for the LLM provider, used for self-hosted
        models.

    Methods:
        to_langchain: Converts the LLM provider's model to a LangChain-compatible model.
        to_instructor: Generates a response using instructor for the selected LLM provider.

    """

    def __init__(
        self,
        model_name: LLMModel,
        api_key: SecretStr,
        # A randomly chosen seed for the model's random generation.
        model_seed: int = 343,
        api_endpoint: str | None = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            model_name (LLMModel): The name of the LLM model to use.
            api_key (SecretStr): The API key for authentication with the LLM provider.
            model_seed (int, optional): Seed for model's random generation. Defaults to 343.
            api_endpoint (str | None, optional): The API endpoint for the LLM provider

        """
        super().__init__(api_key)
        self.model_name = model_name
        self.model_seed = model_seed
        self.api_endpoint = api_endpoint

    @classmethod
    def for_usage(cls, usage: str, config: Config) -> LLMWrapper:
        """Create an LLMWrapper from a LLMModel."""
        model = config.model(usage)
        api_key = config.get_llm_api_key(model)
        api_endpoint = config.get_llm_api_endpoint(model)
        return cls(model, api_key, api_endpoint=api_endpoint)

    def to_langchain(self) -> BaseChatModel:
        """Return a LangChain chat model based on the LLM provider.

        Converts the LLM provider's model to a LangChain-compatible model for interaction
        within the LangChain framework.

        Returns:
            BaseChatModel: A LangChain-compatible model.

        """
        match self.model_name.provider():
            case LLMProvider.OPENAI:
                return ChatOpenAI(
                    name=self.model_name.value,
                    model=self.model_name.value,
                    seed=self.model_seed,
                    api_key=self.api_key,
                    max_retries=3,
                    # Unfortunately you get errors from o3 mini with Langchain unless you set
                    # temperature to 1. See https://github.com/ai-christianson/RA.Aid/issues/70
                    temperature=1 if self.model_name == LLMModel.O_3_MINI else 0,
                    # This is a workaround for o3 mini to avoid parallel tool calls.
                    # See https://github.com/langchain-ai/langchain/issues/25357
                    disabled_params={"parallel_tool_calls": None},
                )
            case LLMProvider.AZURE_OPENAI:
                # Copied settings from OpenAI as the clients and models are the same.
                return AzureChatOpenAI(
                    name=self.model_name.value,
                    model=self.model_name.value,
                    seed=self.model_seed,
                    api_key=self.api_key,
                    max_retries=3,
                    temperature=1 if self.model_name == LLMModel.O_3_MINI else 0,
                    disabled_params={"parallel_tool_calls": None},
                )
            case LLMProvider.ANTHROPIC:
                return ChatAnthropic(
                    model_name=self.model_name.value,
                    timeout=120,
                    stop=None,
                    max_retries=3,
                    api_key=self.api_key,
                )
            case LLMProvider.MISTRALAI:
                validate_extras_dependencies("mistral")
                from langchain_mistralai import ChatMistralAI

                return ChatMistralAI(
                    model_name=self.model_name.value,
                    api_key=self.api_key,
                    max_retries=3,
                )
            case LLMProvider.GOOGLE_GENERATIVE_AI:
                validate_extras_dependencies("google")
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(
                    model=self.model_name.value,
                    api_key=self.api_key,
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
        match self.model_name.provider():
            case LLMProvider.OPENAI:
                client = instructor.from_openai(
                    client=wrappers.wrap_openai(
                        OpenAI(
                            api_key=self.api_key.get_secret_value(),
                        ),
                    ),
                    mode=instructor.Mode.JSON,
                )
                return client.chat.completions.create(
                    response_model=response_model,
                    messages=messages,
                    model=self.model_name.value,
                    seed=self.model_seed,
                )
            case LLMProvider.AZURE_OPENAI:
                if self.api_endpoint is None:
                    raise ValueError("API endpoint is required for Azure OpenAI")
                client = instructor.from_openai(  # pyright: ignore[reportCallIssue]
                    client=AzureOpenAI(  # pyright: ignore[reportArgumentType]
                        api_key=self.api_key.get_secret_value(),
                        azure_endpoint=self.api_endpoint,
                    ),
                    mode=instructor.Mode.JSON,
                )
                return client.chat.completions.create(
                    response_model=response_model,
                    messages=messages,
                    model=self.model_name.value,
                    seed=self.model_seed,
                )
            case LLMProvider.ANTHROPIC:
                client = instructor.from_anthropic(
                    client=wrappers.wrap_anthropic(
                        Anthropic(
                            api_key=self.api_key.get_secret_value(),
                        ),
                    ),
                    mode=instructor.Mode.ANTHROPIC_JSON,
                )
                return client.chat.completions.create(
                    model=self.model_name.value,
                    response_model=response_model,
                    messages=messages,
                    max_tokens=2048,
                )
            case LLMProvider.MISTRALAI:
                validate_extras_dependencies("mistral")
                from mistralai import Mistral

                client = instructor.from_mistral(
                    client=Mistral(
                        api_key=self.api_key.get_secret_value(),
                    ),
                )
                return client.chat.completions.create(  # pyright: ignore[reportReturnType]
                    model=self.model_name.value,
                    response_model=response_model,
                    messages=messages,
                )
            case LLMProvider.GOOGLE_GENERATIVE_AI:
                validate_extras_dependencies("google")
                import google.generativeai as genai

                genai.configure(api_key=self.api_key.get_secret_value()) # pyright: ignore[reportPrivateImportUsage]
                client = instructor.from_gemini(
                    client=genai.GenerativeModel(  # pyright: ignore[reportPrivateImportUsage]
                        model_name=self.model_name.value,
                    ),
                    mode=instructor.Mode.GEMINI_JSON,
                )
                return client.messages.create(  # pyright: ignore[reportReturnType]
                    messages=messages,
                    response_model=response_model,
                )
