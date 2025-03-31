"""Wrapper around different LLM providers, standardizing their usage.

WARNING: This module is deprecated. It will be removed in a future version.

This module provides an abstraction layer around various large language model (LLM) providers,
allowing them to be treated uniformly in the application. It defines an `LLMWrapper` that wraps
a `Model` instance and provides methods to convert the provider's model to LangChain-compatible
models and to generate responses using the instructor tool.

Classes in this file include:

- `LLMWrapper`: A concrete implementation that supports different LLM providers and provides
functionality for converting to LangChain models and generating responses using instructor.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from portia.model import GenerativeModel, LangChainGenerativeModel, Message

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from portia.config import Config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMWrapper:
    """LLMWrapper class for different LLMs.

    This class provides functionality for working with various LLM providers, such as OpenAI,
    Anthropic, and MistralAI. It includes methods to convert the LLM provider's model to a
    LangChain-compatible model and to generate responses using the instructor tool.

    Attributes:
        model_name (LLMModel): The name of the model to use.
        api_key (SecretStr): The API key for the LLM provider.
        model_seed (int): The seed for the model's random generation.
        api_endpoint (str | None): The API endpoint for the LLM provider (Optional, many API's don't
                                   require it).

    Methods:
        to_langchain: Converts the LLM provider's model to a LangChain-compatible model.
        to_instructor: Generates a response using instructor for the selected LLM provider.

    """

    def __init__(
        self,
        model: GenerativeModel,
    ) -> None:
        """Initialize the wrapper.

        Args:
            model (Model): The language model to use.

        """
        self.model = model

    @classmethod
    def for_usage(cls, usage: str, config: Config) -> LLMWrapper:
        """Create an LLMWrapper from a LLMModel."""
        model = config.model(usage)
        return cls(model)

    def to_langchain(self) -> BaseChatModel:
        """Return a LangChain chat model based on the LLM provider.

        Converts the LLM provider's model to a LangChain-compatible model for interaction
        within the LangChain framework.

        Returns:
            BaseChatModel: A LangChain-compatible model.

        """
        if isinstance(self.model, LangChainGenerativeModel):
            return self.model.to_langchain()
        raise ValueError(
            f"LangChain is not supported for this model type {self.model.__class__.__name__}",
        )

    def to_instructor(
        self,
        response_model: type[T],
        messages: list[Message],
    ) -> T:
        """Use instructor to generate an object of the specified response model type.

        Args:
            response_model (type[T]): The Pydantic model to deserialize the response into.
            messages (list[Message]): The messages to send to the LLM.

        Returns:
            T: The deserialized response from the LLM provider.

        """
        return self.model.get_structured_response(messages, response_model)
