"""LLM provider model classes for Portia Agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import instructor
import tiktoken
from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langsmith import wrappers
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, SecretStr, ValidationError

from portia.common import validate_extras_dependencies

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from openai.types.chat import ChatCompletionMessageParam


class Message(BaseModel):
    """Portia LLM message class."""

    role: Literal["user", "assistant", "system"]
    content: str

    @classmethod
    def from_langchain(cls, message: BaseMessage) -> Message:
        """Create a Message from a LangChain message.

        Args:
            message (BaseMessage): The LangChain message to convert.

        Returns:
            Message: The converted message.

        """
        if isinstance(message, HumanMessage):
            return cls.model_validate(
                {"role": "user", "content": message.content or ""},
            )
        if isinstance(message, AIMessage):
            return cls.model_validate(
                {"role": "assistant", "content": message.content or ""},
            )
        if isinstance(message, SystemMessage):
            return cls.model_validate(
                {"role": "system", "content": message.content or ""},
            )
        raise ValueError(f"Unsupported message type: {type(message)}")

    def to_langchain(self) -> BaseMessage:
        """Convert to LangChain BaseMessage sub-type.

        Returns:
            BaseMessage: The converted message, subclass of LangChain's BaseMessage.

        """
        if self.role == "user":
            return HumanMessage(content=self.content)
        if self.role == "assistant":
            return AIMessage(content=self.content)
        if self.role == "system":
            return SystemMessage(content=self.content)
        raise ValueError(f"Unsupported role: {self.role}")


class LLMProvider(Enum):
    """Enum for supported LLM providers.

    Attributes:
        OPENAI: OpenAI provider.
        ANTHROPIC: Anthropic provider.
        MISTRALAI: MistralAI provider.
        GOOGLE_GENERATIVE_AI: Google Generative AI provider.
        AZURE_OPENAI: Azure OpenAI provider.

    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRALAI = "mistralai"
    GOOGLE_GENERATIVE_AI = "google"
    AZURE_OPENAI = "azure-openai"
    CUSTOM = "custom"
    OLLAMA = "ollama"


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


class GenerativeModel(ABC):
    """Base class for all generative model clients."""

    provider: LLMProvider

    def __init__(self, model_name: str) -> None:
        """Initialize the model.

        Args:
            model_name: The name of the model.

        """
        self.model_name = model_name

    @abstractmethod
    def get_response(self, messages: list[Message]) -> Message:
        """Given a list of messages, call the model and return its response as a new message.

        Args:
            messages (list[Message]): The list of messages to send to the model.

        Returns:
            Message: The response from the model.

        """

    @abstractmethod
    def get_structured_response(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
    ) -> BaseModelT:
        """Get a structured response from the model, given a Pydantic model.

        Args:
            messages (list[Message]): The list of messages to send to the model.
            schema (type[BaseModelT]): The Pydantic model to use for the response.

        Returns:
            BaseModelT: The structured response from the model.

        """

    def __str__(self) -> str:
        """Get the string representation of the model."""
        return f"{self.provider.value}/{self.model_name}"

    def __repr__(self) -> str:
        """Get the string representation of the model."""
        return f'{self.__class__.__name__}("{self.provider.value}/{self.model_name}")'

    @abstractmethod
    def to_langchain(self) -> BaseChatModel:
        """Get the LangChain client."""


class LangChainGenerativeModel(GenerativeModel):
    """Base class for LangChain-based models."""

    provider: LLMProvider = LLMProvider.CUSTOM

    def __init__(self, client: BaseChatModel, model_name: str) -> None:
        """Initialize with LangChain client.

        Args:
            client: LangChain chat model instance
            model_name: The name of the model

        """
        super().__init__(model_name)
        self._client = client

    def to_langchain(self) -> BaseChatModel:
        """Get the LangChain client."""
        return self._client

    def get_response(self, messages: list[Message]) -> Message:
        """Get response using LangChain model."""
        langchain_messages = [msg.to_langchain() for msg in messages]
        response = self._client.invoke(langchain_messages)
        return Message.from_langchain(response)

    def get_structured_response(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
        **kwargs: Any,
    ) -> BaseModelT:
        """Get structured response using LangChain model.

        Args:
            messages (list[Message]): The list of messages to send to the model.
            schema (type[BaseModelT]): The Pydantic model to use for the response.
            **kwargs: Additional keyword arguments to pass to the with_structured_output method.

        Returns:
            BaseModelT: The structured response from the model.

        """
        langchain_messages = [msg.to_langchain() for msg in messages]
        structured_client = self._client.with_structured_output(schema, **kwargs)
        response = structured_client.invoke(langchain_messages)
        if isinstance(response, schema):
            return response
        return schema.model_validate(response)


class OpenAIGenerativeModel(LangChainGenerativeModel):
    """OpenAI model implementation."""

    provider: LLMProvider = LLMProvider.OPENAI

    def __init__(
        self,
        *,
        model_name: str,
        api_key: SecretStr,
        seed: int = 343,
        max_retries: int = 3,
        temperature: float = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize with OpenAI client.

        Args:
            model_name: OpenAI model to use
            api_key: API key for OpenAI
            seed: Random seed for model generation
            max_retries: Maximum number of retries
            temperature: Temperature parameter
            **kwargs: Additional keyword arguments to pass to ChatOpenAI

        """
        if "disabled_params" not in kwargs:
            # This is a workaround for o3 mini to avoid parallel tool calls.
            # See https://github.com/langchain-ai/langchain/issues/25357
            kwargs["disabled_params"] = {"parallel_tool_calls": None}

        # Unfortunately you get errors from o3 mini with Langchain unless you set
        # temperature to 1. See https://github.com/ai-christianson/RA.Aid/issues/70
        temperature = 1 if "o3-mini" in model_name.lower() else temperature

        client = ChatOpenAI(
            name=model_name,
            model=model_name,
            seed=seed,
            api_key=api_key,
            max_retries=max_retries,
            temperature=temperature,
            **kwargs,
        )
        super().__init__(client, model_name)
        self._instructor_client = instructor.from_openai(
            client=wrappers.wrap_openai(OpenAI(api_key=api_key.get_secret_value())),
            mode=instructor.Mode.JSON,
        )
        self._seed = seed

    def get_structured_response(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
        **kwargs: Any,
    ) -> BaseModelT:
        """Call the model in structured output mode targeting the given Pydantic model.

        Args:
            messages (list[Message]): The list of messages to send to the model.
            schema (type[BaseModelT]): The Pydantic model to use for the response.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            BaseModelT: The structured response from the model.

        """
        if schema.__name__ == "StepsOrError":
            return self.get_structured_response_instructor(messages, schema)
        return super().get_structured_response(
            messages,
            schema,
            method="function_calling",
            **kwargs,
        )

    def get_structured_response_instructor(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
    ) -> BaseModelT:
        """Get structured response using instructor."""
        instructor_messages = [map_message_to_instructor(msg) for msg in messages]
        return self._instructor_client.chat.completions.create(
            response_model=schema,
            messages=instructor_messages,
            model=self.model_name,
            seed=self._seed,
        )


class AzureOpenAIGenerativeModel(LangChainGenerativeModel):
    """Azure OpenAI model implementation."""

    provider: LLMProvider = LLMProvider.AZURE_OPENAI

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_name: str,
        api_key: SecretStr,
        azure_endpoint: str,
        api_version: str = "2025-01-01-preview",
        seed: int = 343,
        max_retries: int = 3,
        temperature: float = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize with Azure OpenAI client.

        Args:
            model_name: OpenAI model to use
            azure_endpoint: Azure OpenAI endpoint
            api_version: Azure API version
            seed: Random seed for model generation
            api_key: API key for Azure OpenAI
            max_retries: Maximum number of retries
            temperature: Temperature parameter (defaults to 1 for O_3_MINI, 0 otherwise)
            **kwargs: Additional keyword arguments to pass to AzureChatOpenAI

        """
        if "disabled_params" not in kwargs:
            # This is a workaround for o3 mini to avoid parallel tool calls.
            # See https://github.com/langchain-ai/langchain/issues/25357
            kwargs["disabled_params"] = {"parallel_tool_calls": None}

        # Unfortunately you get errors from o3 mini with Langchain unless you set
        # temperature to 1. See https://github.com/ai-christianson/RA.Aid/issues/70
        temperature = 1 if "o3-mini" in model_name.lower() else temperature

        client = AzureChatOpenAI(
            name=model_name,
            model=model_name,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            seed=seed,
            api_key=api_key,
            max_retries=max_retries,
            temperature=temperature,
            **kwargs,
        )
        super().__init__(client, model_name)
        self._instructor_client = instructor.from_openai(
            client=AzureOpenAI(
                api_key=api_key.get_secret_value(),
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            ),
            mode=instructor.Mode.JSON,
        )
        self._seed = seed

    def get_structured_response(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
        **kwargs: Any,
    ) -> BaseModelT:
        """Call the model in structured output mode targeting the given Pydantic model.

        Args:
            messages (list[Message]): The list of messages to send to the model.
            schema (type[BaseModelT]): The Pydantic model to use for the response.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            BaseModelT: The structured response from the model.

        """
        if schema.__name__ == "StepsOrError":
            return self.get_structured_response_instructor(messages, schema)
        return super().get_structured_response(
            messages,
            schema,
            method="function_calling",
            **kwargs,
        )

    def get_structured_response_instructor(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
    ) -> BaseModelT:
        """Get structured response using instructor."""
        instructor_messages = [map_message_to_instructor(msg) for msg in messages]
        return self._instructor_client.chat.completions.create(
            response_model=schema,
            messages=instructor_messages,
            model=self.model_name,
            seed=self._seed,
        )


class AnthropicGenerativeModel(LangChainGenerativeModel):
    """Anthropic model implementation."""

    provider: LLMProvider = LLMProvider.ANTHROPIC
    _output_instructor_threshold = 512

    def __init__(
        self,
        *,
        model_name: str = "claude-3-5-sonnet-latest",
        api_key: SecretStr,
        timeout: int = 120,
        max_retries: int = 3,
        max_tokens: int = 8096,
        **kwargs: Any,
    ) -> None:
        """Initialize with Anthropic client.

        Args:
            model_name: Name of the Anthropic model
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            max_tokens: Maximum number of tokens to generate
            api_key: API key for Anthropic
            **kwargs: Additional keyword arguments to pass to ChatAnthropic

        """
        client = ChatAnthropic(
            model_name=model_name,
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,  # pyright: ignore[reportCallIssue]
            api_key=api_key,
            **kwargs,
        )
        super().__init__(client, model_name)
        self._instructor_client = instructor.from_anthropic(
            client=wrappers.wrap_anthropic(
                Anthropic(api_key=api_key.get_secret_value()),
            ),
            mode=instructor.Mode.ANTHROPIC_JSON,
        )
        self.max_tokens = max_tokens

    def get_structured_response(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
        **kwargs: Any,
    ) -> BaseModelT:
        """Call the model in structured output mode targeting the given Pydantic model.

        Args:
            messages (list[Message]): The list of messages to send to the model.
            schema (type[BaseModelT]): The Pydantic model to use for the response.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            BaseModelT: The structured response from the model.

        """
        if schema.__name__ == "StepsOrError":
            return self.get_structured_response_instructor(messages, schema)
        langchain_messages = [msg.to_langchain() for msg in messages]
        structured_client = self._client.with_structured_output(schema, include_raw=True, **kwargs)
        raw_response = structured_client.invoke(langchain_messages)
        if not isinstance(raw_response, dict):
            raise TypeError(f"Expected dict, got {type(raw_response).__name__}.")
        # Anthropic sometimes struggles serializing large JSON responses, so we fall back to
        # instructor if the response is above a certain size.
        if isinstance(raw_response.get("parsing_error"), ValidationError) and (
            len(tiktoken.get_encoding("gpt2").encode(raw_response["raw"].model_dump_json()))
            > self._output_instructor_threshold
        ):
            return self.get_structured_response_instructor(messages, schema)
        response = raw_response["parsed"]
        if isinstance(response, schema):
            return response
        return schema.model_validate(response)

    def get_structured_response_instructor(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
    ) -> BaseModelT:
        """Get structured response using instructor."""
        instructor_messages = [map_message_to_instructor(msg) for msg in messages]
        return self._instructor_client.chat.completions.create(
            model=self.model_name,
            response_model=schema,
            messages=instructor_messages,
            max_tokens=self.max_tokens,
        )


if validate_extras_dependencies("mistralai", raise_error=False):
    from langchain_mistralai import ChatMistralAI
    from mistralai import Mistral

    class MistralAIGenerativeModel(LangChainGenerativeModel):
        """MistralAI model implementation."""

        provider: LLMProvider = LLMProvider.MISTRALAI

        def __init__(
            self,
            *,
            model_name: str = "mistral-large-latest",
            api_key: SecretStr,
            max_retries: int = 3,
            **kwargs: Any,
        ) -> None:
            """Initialize with MistralAI client.

            Args:
                model_name: Name of the MistralAI model
                api_key: API key for MistralAI
                max_retries: Maximum number of retries
                **kwargs: Additional keyword arguments to pass to ChatMistralAI

            """
            client = ChatMistralAI(
                model_name=model_name,
                api_key=api_key,
                max_retries=max_retries,
                **kwargs,
            )
            super().__init__(client, model_name)
            self._instructor_client = instructor.from_mistral(
                client=Mistral(api_key=api_key.get_secret_value()),
                use_async=False,
            )

        def get_structured_response(
            self,
            messages: list[Message],
            schema: type[BaseModelT],
            **kwargs: Any,
        ) -> BaseModelT:
            """Call the model in structured output mode targeting the given Pydantic model.

            Args:
                messages (list[Message]): The list of messages to send to the model.
                schema (type[BaseModelT]): The Pydantic model to use for the response.
                **kwargs: Additional keyword arguments to pass to the model.

            Returns:
                BaseModelT: The structured response from the model.

            """
            if schema.__name__ == "StepsOrError":
                return self.get_structured_response_instructor(messages, schema)
            return super().get_structured_response(
                messages,
                schema,
                method="function_calling",
                **kwargs,
            )

        def get_structured_response_instructor(
            self,
            messages: list[Message],
            schema: type[BaseModelT],
        ) -> BaseModelT:
            """Get structured response using instructor."""
            instructor_messages = [map_message_to_instructor(msg) for msg in messages]
            return self._instructor_client.chat.completions.create(
                model=self.model_name,
                response_model=schema,
                messages=instructor_messages,
            )


if validate_extras_dependencies("google", raise_error=False):
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI

    if TYPE_CHECKING:
        from google.generativeai.types.generation_types import GenerationConfigDict

    class GoogleGenAiGenerativeModel(LangChainGenerativeModel):
        """Google Generative AI (Gemini)model implementation."""

        provider: LLMProvider = LLMProvider.GOOGLE_GENERATIVE_AI

        def __init__(
            self,
            *,
            model_name: str = "gemini-2.0-flash",
            api_key: SecretStr,
            max_retries: int = 3,
            temperature: float | None = None,
            **kwargs: Any,
        ) -> None:
            """Initialize with Google Generative AI client.

            Args:
                model_name: Name of the Google Generative AI model
                api_key: API key for Google Generative AI
                max_retries: Maximum number of retries
                temperature: Temperature parameter for model sampling
                **kwargs: Additional keyword arguments to pass to ChatGoogleGenerativeAI

            """
            # Configure genai with the api key
            genai.configure(api_key=api_key.get_secret_value())  # pyright: ignore[reportPrivateImportUsage]

            generation_config: GenerationConfigDict = {}
            if temperature is not None:
                kwargs["temperature"] = temperature
                generation_config["temperature"] = temperature

            client = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=api_key,
                max_retries=max_retries,
                **kwargs,
            )
            super().__init__(client, model_name)
            self._instructor_client = instructor.from_gemini(
                client=genai.GenerativeModel(  # pyright: ignore[reportPrivateImportUsage]
                    model_name=model_name,
                    generation_config=generation_config,
                ),
                mode=instructor.Mode.GEMINI_JSON,
                use_async=False,
            )

        def get_structured_response(
            self,
            messages: list[Message],
            schema: type[BaseModelT],
            **_: Any,
        ) -> BaseModelT:
            """Get structured response from Google Generative AI model using instructor.

            NB. We use the instructor library to get the structured response, because the Google
            Generative AI API does not support Any-types in structured output mode. Instructor
            works around this by NOT using the API structured output mode, and instead using the
            text generation API to generate a JSON-formatted response, which is then parsed into
            the Pydantic model.

            Args:
                messages (list[Message]): The list of messages to send to the model.
                schema (type[BaseModelT]): The Pydantic model to use for the response.
                **kwargs: Additional keyword arguments to pass to the model.

            Returns:
                BaseModelT: The structured response from the model.

            """
            instructor_messages = [map_message_to_instructor(msg) for msg in messages]
            return self._instructor_client.messages.create(
                messages=instructor_messages,
                response_model=schema,
            )


if validate_extras_dependencies("ollama", raise_error=False):
    from langchain_ollama import ChatOllama

    class OllamaGenerativeModel(LangChainGenerativeModel):
        """Wrapper for Ollama models."""

        provider_name: str = "ollama"

        def __init__(
            self,
            model_name: str,
            base_url: str = "http://localhost:11434/v1",
            **kwargs: Any,
        ) -> None:
            """Initialize with Ollama client.

            Args:
                model_name: Name of the Ollama model
                base_url: Base URL of the Ollama server
                **kwargs: Additional keyword arguments to pass to ChatOllama

            """
            super().__init__(
                client=ChatOllama(model=model_name, **kwargs),
                model_name=model_name,
            )
            self.base_url = base_url

        def get_structured_response(
            self,
            messages: list[Message],
            schema: type[BaseModelT],
            **kwargs: Any,  # noqa: ARG002
        ) -> BaseModelT:
            """Get structured response from Ollama model using instructor.

            Args:
                messages (list[Message]): The list of messages to send to the model.
                schema (type[BaseModelT]): The Pydantic model to use for the response.
                **kwargs: Additional keyword arguments to pass to the model.

            Returns:
                BaseModelT: The structured response from the model.

            """
            client = instructor.from_openai(
                OpenAI(
                    base_url=self.base_url,
                    api_key="ollama",  # required, but unused
                ),
                mode=instructor.Mode.JSON,
            )
            return client.chat.completions.create(
                model=self.model_name,
                messages=[map_message_to_instructor(message) for message in messages],
                response_model=schema,
                max_retries=2,
            )


def map_message_to_instructor(message: Message) -> ChatCompletionMessageParam:
    """Map a Message to ChatCompletionMessageParam.

    Args:
        message (Message): The message to map.

    Returns:
        ChatCompletionMessageParam: Message in the format expected by instructor.

    """
    match message:
        case Message(role="user", content=content):
            return {"role": "user", "content": content}
        case Message(role="assistant", content=content):
            return {"role": "assistant", "content": content}
        case Message(role="system", content=content):
            return {"role": "system", "content": content}
        case _:
            raise ValueError(f"Unsupported message role: {message.role}")
