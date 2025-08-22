"""LLM provider model classes for Portia Agents."""

from __future__ import annotations

import copy
import hashlib
import json
from abc import ABC, abstractmethod
from contextlib import suppress
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import instructor
import litellm
from anthropic import Anthropic, AsyncAnthropic
from langchain.globals import set_llm_cache
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import Generation
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langsmith import wrappers
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from pydantic import BaseModel, SecretStr, ValidationError
from redis import RedisError

from portia.common import validate_extras_dependencies
from portia.logger import logger
from portia.token_check import estimate_tokens

if TYPE_CHECKING:
    from langchain_core.caches import BaseCache
    from langchain_core.language_models.chat_models import BaseChatModel
    from openai.types.chat import ChatCompletionMessageParam

_llm_cache: ContextVar[BaseCache | None] = ContextVar("llm_cache", default=None)


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
        GOOGLE: Google Generative AI provider.
        AZURE_OPENAI: Azure OpenAI provider.
        GROQ: Groq provider.

    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRALAI = "mistralai"
    GOOGLE = "google"
    AMAZON = "amazon"
    AZURE_OPENAI = "azure-openai"
    CUSTOM = "custom"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    GOOGLE_GENERATIVE_AI = "google"  # noqa: PIE796 - Alias for GOOGLE member


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

    def _log_llm_call(self, messages: list[Message]) -> None:
        """Log TRACE level information about the LLM call."""
        with suppress(Exception):
            if messages:
                content_preview = " ".join(msg.content.replace("\n", " ") for msg in messages)
                preview = content_preview[:120]
                logger().trace(f"LLM call: model={self!s} msg={preview!r}")

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

    @abstractmethod
    async def aget_response(self, messages: list[Message]) -> Message:
        """Given a list of messages, call the model and return its response as a new message async.

        Args:
            messages (list[Message]): The list of messages to send to the model.

        """
        raise NotImplementedError("async is not implemented")  # pragma: no cover

    @abstractmethod
    async def aget_structured_response(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
    ) -> BaseModelT:
        """Get a structured response from the model, given a Pydantic model asynchronously.

        Args:
            messages (list[Message]): The list of messages to send to the model.
            schema (type[BaseModelT]): The Pydantic model to use for the response.

        """
        raise NotImplementedError("async is not implemented")  # pragma: no cover

    def get_context_window_size(self) -> int:
        """Get the context window size of the model.

        Falls back to 100k tokens if the model is not found.
        """
        try:
            return litellm.model_cost[self.model_name]["max_input_tokens"]
        except (KeyError, TypeError):
            return 100_000

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

    def __init__(
        self,
        client: BaseChatModel,
        model_name: str,
    ) -> None:
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
        super()._log_llm_call(messages)
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
        super()._log_llm_call(messages)
        langchain_messages = [msg.to_langchain() for msg in messages]
        structured_client = self._client.with_structured_output(schema, **kwargs)
        response = structured_client.invoke(langchain_messages)
        if isinstance(response, schema):
            return response
        return schema.model_validate(response)

    async def aget_response(self, messages: list[Message]) -> Message:
        """Get response using LangChain model asynchronously.

        Args:
            messages (list[Message]): The list of messages to send to the model.

        """
        super()._log_llm_call(messages)
        langchain_messages = [msg.to_langchain() for msg in messages]
        response = await self._client.ainvoke(langchain_messages)
        return Message.from_langchain(response)

    async def aget_structured_response(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
        **kwargs: Any,
    ) -> BaseModelT:
        """Get structured response using LangChain model asynchronously.

        Args:
            messages (list[Message]): The list of messages to send to the model.
            schema (type[BaseModelT]): The Pydantic model to use for the response.
            **kwargs: Additional keyword arguments to pass to the with_structured_output method.

        """
        super()._log_llm_call(messages)
        langchain_messages = [msg.to_langchain() for msg in messages]
        structured_client = self._client.with_structured_output(schema, **kwargs)
        response = await structured_client.ainvoke(langchain_messages)
        if isinstance(response, schema):
            return response
        return schema.model_validate(response)

    def _cached_instructor_call(
        self,
        client: instructor.Instructor,
        messages: list[ChatCompletionMessageParam],
        schema: type[BaseModelT],
        provider: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> BaseModelT:
        """Call an instructor client with caching enabled if it is set up."""
        if model is not None:
            kwargs["model"] = model

        cache = _llm_cache.get()
        if cache is None:
            return client.chat.completions.create(
                response_model=schema, messages=messages, **kwargs
            )

        cache_data = {
            "schema": schema.model_json_schema(),
            **kwargs,
        }
        data_hash = hashlib.md5(  # nosec B324  # noqa: S324
            json.dumps(cache_data, sort_keys=True, default=str).encode()
        ).hexdigest()
        llm_string = f"{provider}:{model}:{data_hash}"
        prompt = json.dumps(messages)
        try:
            cached = cache.lookup(prompt, llm_string)
            if cached and len(cached) > 0:
                return schema.model_validate_json(cached[0].text)  # pyright: ignore[reportArgumentType]
        except (ValidationError, RedisError, AttributeError):
            # On validation errors, re-fetch and update the entry in the cache
            pass
        response = client.chat.completions.create(
            response_model=schema, messages=messages, **kwargs
        )
        cache.update(prompt, llm_string, [Generation(text=response.model_dump_json())])
        return response

    async def _acached_instructor_call(
        self,
        client: instructor.AsyncInstructor,
        messages: list[ChatCompletionMessageParam],
        schema: type[BaseModelT],
        provider: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> BaseModelT:
        """Call an instructor client with caching enabled if it is set up asynchronously."""
        if model is not None:
            kwargs["model"] = model

        cache = _llm_cache.get()
        if cache is None:
            return await client.chat.completions.create(
                response_model=schema, messages=messages, **kwargs
            )

        cache_data = {
            "schema": schema.model_json_schema(),
            **kwargs,
        }
        data_hash = hashlib.md5(  # nosec B324  # noqa: S324
            json.dumps(cache_data, sort_keys=True, default=str).encode()
        ).hexdigest()
        llm_string = f"{provider}:{model}:{data_hash}"
        prompt = json.dumps(messages)
        try:
            cached = await cache.alookup(prompt, llm_string)
            if cached and len(cached) > 0:
                return schema.model_validate_json(cached[0].text)  # pyright: ignore[reportArgumentType]
        except (ValidationError, RedisError, AttributeError):
            # On validation errors, re-fetch and update the entry in the cache
            pass
        response = await client.chat.completions.create(
            response_model=schema, messages=messages, **kwargs
        )
        await cache.aupdate(prompt, llm_string, [Generation(text=response.model_dump_json())])
        return response

    @classmethod
    def set_cache(cls, cache: BaseCache) -> None:
        """Set the cache for the model."""
        _llm_cache.set(cache)
        set_llm_cache(cache)


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
        self._model_kwargs = kwargs.copy()

        if "disabled_params" not in kwargs:
            # This is a workaround for o3 mini to avoid parallel tool calls.
            # See https://github.com/langchain-ai/langchain/issues/25357
            kwargs["disabled_params"] = {"parallel_tool_calls": None}

        # Unfortunately you get errors from o3 mini with Langchain unless you set
        # temperature to 1. See https://github.com/ai-christianson/RA.Aid/issues/70
        temperature = 1 if model_name.lower() in ("o3-mini", "o4-mini", "gpt-5") else temperature

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
        self._instructor_client_async = instructor.from_openai(
            client=wrappers.wrap_openai(AsyncOpenAI(api_key=api_key.get_secret_value())),
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
        if schema.__name__ in ("StepsOrError", "PreStepIntrospection"):
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
        super()._log_llm_call(messages)
        instructor_messages = [map_message_to_instructor(msg) for msg in messages]
        return self._cached_instructor_call(
            client=self._instructor_client,
            messages=instructor_messages,
            schema=schema,
            model=self.model_name,
            provider=self.provider.value,
            seed=self._seed,
            **self._model_kwargs,
        )

    async def aget_structured_response(
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
        if schema.__name__ in ("StepsOrError", "PreStepIntrospection"):
            return await self.aget_structured_response_instructor(messages, schema)

        return await super().aget_structured_response(
            messages,
            schema,
            method="function_calling",
            **kwargs,
        )

    async def aget_structured_response_instructor(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
    ) -> BaseModelT:
        """Get structured response using instructor asynchronously."""
        super()._log_llm_call(messages)
        instructor_messages = [map_message_to_instructor(msg) for msg in messages]
        return await self._acached_instructor_call(
            client=self._instructor_client_async,
            messages=instructor_messages,
            schema=schema,
            model=self.model_name,
            provider=self.provider.value,
            seed=self._seed,
            **self._model_kwargs,
        )


class OpenRouterGenerativeModel(OpenAIGenerativeModel):
    """OpenRouter model implementation."""

    provider: LLMProvider = LLMProvider.OPENROUTER

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
        """Initialize with OpenRouter client.

        Args:
            model_name: OpenRouter model to use
            api_key: API key for OpenRouter
            seed: Random seed for model generation
            max_retries: Maximum number of retries
            temperature: Temperature parameter
            **kwargs: Additional keyword arguments to pass to ChatOpenAI

        """
        self._model_kwargs = kwargs.copy()
        if "disabled_params" not in kwargs:
            # This is a workaround for o3 mini to avoid parallel tool calls.
            # See https://github.com/langchain-ai/langchain/issues/25357
            kwargs["disabled_params"] = {"parallel_tool_calls": None}
        # Unfortunately you get errors from o3 mini with Langchain unless you set
        # temperature to 1. See https://github.com/ai-christianson/RA.Aid/issues/70
        temperature = 1 if model_name.lower() in ("o3-mini", "o4-mini", "gpt-5") else temperature

        # OpenRouter is compatible with the ChatOpenAI client, so we use this client
        # with the openrouter URL
        client = ChatOpenAI(
            name=model_name,
            model=model_name,
            seed=seed,
            api_key=api_key,
            max_retries=max_retries,
            temperature=temperature,
            base_url="https://openrouter.ai/api/v1",
            **kwargs,
        )
        super(OpenAIGenerativeModel, self).__init__(client, model_name)
        self._instructor_client = instructor.from_openai(
            client=wrappers.wrap_openai(
                OpenAI(api_key=api_key.get_secret_value(), base_url="https://openrouter.ai/api/v1")
            ),
            mode=instructor.Mode.JSON,
        )
        self._instructor_client_async = instructor.from_openai(
            client=wrappers.wrap_openai(
                AsyncOpenAI(
                    api_key=api_key.get_secret_value(), base_url="https://openrouter.ai/api/v1"
                )
            ),
            mode=instructor.Mode.JSON,
        )
        self._seed = seed


class GroqGenerativeModel(OpenAIGenerativeModel):
    """Groq model implementation."""

    provider: LLMProvider = LLMProvider.GROQ

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
        """Initialize with Groq client.

        Args:
            model_name: Groq model to use
            api_key: API key for Groq
            seed: Random seed for model generation
            max_retries: Maximum number of retries
            temperature: Temperature parameter
            **kwargs: Additional keyword arguments to pass to ChatOpenAI

        """
        self._model_kwargs = kwargs.copy()
        if "disabled_params" not in kwargs:
            # This is a workaround for some models to avoid parallel tool calls.
            # See https://github.com/langchain-ai/langchain/issues/25357
            kwargs["disabled_params"] = {"parallel_tool_calls": None}

        # Groq is compatible with the ChatOpenAI client, so we use this client
        # with the groq URL
        client = ChatOpenAI(
            name=model_name,
            model=model_name,
            seed=seed,
            api_key=api_key,
            max_retries=max_retries,
            temperature=temperature,
            base_url="https://api.groq.com/openai/v1",
            **kwargs,
        )
        super(OpenAIGenerativeModel, self).__init__(client, model_name)
        self._instructor_client = instructor.from_openai(
            client=wrappers.wrap_openai(
                OpenAI(api_key=api_key.get_secret_value(), base_url="https://api.groq.com/openai/v1")
            ),
            mode=instructor.Mode.JSON,
        )
        self._instructor_client_async = instructor.from_openai(
            client=wrappers.wrap_openai(
                AsyncOpenAI(
                    api_key=api_key.get_secret_value(), base_url="https://api.groq.com/openai/v1"
                )
            ),
            mode=instructor.Mode.JSON,
        )
        self._seed = seed


class AzureOpenAIGenerativeModel(LangChainGenerativeModel):
    """Azure OpenAI model implementation."""

    provider: LLMProvider = LLMProvider.AZURE_OPENAI

    def __init__(
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
        self._model_kwargs = kwargs.copy()

        if "disabled_params" not in kwargs:
            # This is a workaround for o3 mini to avoid parallel tool calls.
            # See https://github.com/langchain-ai/langchain/issues/25357
            kwargs["disabled_params"] = {"parallel_tool_calls": None}

        # Unfortunately you get errors from o3/o4 mini with Langchain unless you set
        # temperature to 1. See https://github.com/ai-christianson/RA.Aid/issues/70
        temperature = 1 if model_name.lower() in ("o3-mini", "o4-mini") else temperature

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
        self._instructor_client_async = instructor.from_openai(
            AsyncAzureOpenAI(
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
        super()._log_llm_call(messages)
        instructor_messages = [map_message_to_instructor(msg) for msg in messages]
        return self._cached_instructor_call(
            client=self._instructor_client,
            messages=instructor_messages,
            schema=schema,
            model=self.model_name,
            provider=self.provider.value,
            seed=self._seed,
            **self._model_kwargs,
        )

    async def aget_structured_response(
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
        return await super().aget_structured_response(
            messages,
            schema,
            method="function_calling",
            **kwargs,
        )


class AnthropicGenerativeModel(LangChainGenerativeModel):
    """Anthropic model implementation."""

    provider: LLMProvider = LLMProvider.ANTHROPIC
    _output_instructor_threshold = 512

    def __init__(
        self,
        *,
        model_name: str = "claude-3-7-sonnet-latest",
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
        if "model_kwargs" in kwargs:
            self._model_kwargs = kwargs["model_kwargs"].copy()
        else:
            self._model_kwargs = kwargs.copy()
        client = ChatAnthropic(
            model_name=model_name,
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,  # pyright: ignore[reportCallIssue]
            api_key=api_key,
            **kwargs,
        )
        kwargs_no_thinking = copy.deepcopy(kwargs)
        kwargs_no_thinking.get("model_kwargs", {}).pop("thinking", None)
        # You cannot use structured output with thinking enabled, or you get an error saying
        # 'Thinking may not be enabled when tool_choice forces tool use'.
        # So we create a separate client for structured output.
        # NB Instructor can be used, because it doesn't use the tool_choice API.
        # See https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-thinking-with-tool-use
        self._non_thinking_client = ChatAnthropic(
            model_name=model_name,
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,  # pyright: ignore[reportCallIssue]
            api_key=api_key,
            **kwargs_no_thinking,
        )
        super().__init__(client, model_name)

        self._instructor_client = instructor.from_anthropic(
            client=wrappers.wrap_anthropic(
                Anthropic(api_key=api_key.get_secret_value()),
            ),
            mode=instructor.Mode.ANTHROPIC_JSON,
        )
        self._instructor_client_async = instructor.from_anthropic(
            client=wrappers.wrap_anthropic(
                AsyncAnthropic(api_key=api_key.get_secret_value()),
            ),
            mode=instructor.Mode.ANTHROPIC_JSON,
        )
        self.max_tokens = max_tokens

    def get_response(self, messages: list[Message]) -> Message:
        """Get response from Anthropic model, handling list content."""
        langchain_messages = [msg.to_langchain() for msg in messages]
        response = self._client.invoke(langchain_messages)

        if isinstance(response, AIMessage):
            if isinstance(response.content, list):
                # This is to extract the result from response of anthropic thinking models.
                content = ", ".join(
                    item.get("text", "")
                    for item in response.content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            else:
                content = response.content
            return Message.model_validate(
                {"role": "assistant", "content": content or ""},
            )
        return Message.from_langchain(response)

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
        if schema.__name__ in ("StepsOrError", "PreStepIntrospection"):
            return self.get_structured_response_instructor(messages, schema)
        langchain_messages = [msg.to_langchain() for msg in messages]
        structured_client = self._non_thinking_client.with_structured_output(
            schema,
            include_raw=True,
            **kwargs,
        )
        raw_response = structured_client.invoke(langchain_messages)
        if not isinstance(raw_response, dict):
            raise TypeError(f"Expected dict, got {type(raw_response).__name__}.")
        # Anthropic sometimes struggles serializing large JSON responses, so we fall back to
        # instructor if the response is above a certain size.
        if isinstance(raw_response.get("parsing_error"), ValidationError) and (
            estimate_tokens(raw_response["raw"].model_dump_json())
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
        super()._log_llm_call(messages)
        instructor_messages = [map_message_to_instructor(msg) for msg in messages]
        return self._cached_instructor_call(
            client=self._instructor_client,
            messages=instructor_messages,
            schema=schema,
            model=self.model_name,
            provider=self.provider.value,
            max_tokens=self.max_tokens,
            **self._model_kwargs,
        )

    async def aget_response(self, messages: list[Message]) -> Message:
        """Get response from Anthropic model asynchronously, handling list content."""
        langchain_messages = [msg.to_langchain() for msg in messages]
        response = await self._client.ainvoke(langchain_messages)

        if isinstance(response, AIMessage):
            if isinstance(response.content, list):
                # This is to extract the result from response of anthropic thinking models.
                content = ", ".join(
                    item.get("text", "")
                    for item in response.content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            else:
                content = response.content
            return Message.model_validate(
                {"role": "assistant", "content": content or ""},
            )
        return Message.from_langchain(response)

    async def aget_structured_response(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
        **kwargs: Any,
    ) -> BaseModelT:
        """Call the model in structured output mode targeting the given Pydantic model async.

        Args:
            messages (list[Message]): The list of messages to send to the model.
            schema (type[BaseModelT]): The Pydantic model to use for the response.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            BaseModelT: The structured response from the model.

        """
        if schema.__name__ in ("StepsOrError", "PreStepIntrospection"):
            return await self.aget_structured_response_instructor(messages, schema)
        langchain_messages = [msg.to_langchain() for msg in messages]
        structured_client = self._non_thinking_client.with_structured_output(
            schema,
            include_raw=True,
            **kwargs,
        )
        raw_response = await structured_client.ainvoke(langchain_messages)
        if not isinstance(raw_response, dict):
            raise TypeError(f"Expected dict, got {type(raw_response).__name__}.")
        # Anthropic sometimes struggles serializing large JSON responses, so we fall back to
        # instructor if the response is above a certain size.
        if isinstance(raw_response.get("parsing_error"), ValidationError) and (
            estimate_tokens(raw_response["raw"].model_dump_json())
            > self._output_instructor_threshold
        ):
            return await self.aget_structured_response_instructor(messages, schema)
        response = raw_response["parsed"]
        if isinstance(response, schema):
            return response
        return schema.model_validate(response)

    async def aget_structured_response_instructor(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
    ) -> BaseModelT:
        """Get structured response using instructor asynchronously."""
        super()._log_llm_call(messages)
        instructor_messages = [map_message_to_instructor(msg) for msg in messages]
        return await self._acached_instructor_call(
            client=self._instructor_client_async,
            messages=instructor_messages,
            schema=schema,
            model=self.model_name,
            provider=self.provider.value,
            max_tokens=self.max_tokens,
            **self._model_kwargs,
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
            self._instructor_client_async = instructor.from_mistral(
                client=Mistral(api_key=api_key.get_secret_value()),
                use_async=True,
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
            super()._log_llm_call(messages)
            instructor_messages = [map_message_to_instructor(msg) for msg in messages]
            return self._cached_instructor_call(
                client=self._instructor_client,
                messages=instructor_messages,
                schema=schema,
                model=self.model_name,
                provider=self.provider.value,
            )

        async def aget_response(self, messages: list[Message]) -> Message:
            """Get response from MistralAI model asynchronously."""
            langchain_messages = [msg.to_langchain() for msg in messages]
            response = await self._client.ainvoke(langchain_messages)
            return Message.from_langchain(response)

        async def aget_structured_response(
            self,
            messages: list[Message],
            schema: type[BaseModelT],
            **kwargs: Any,
        ) -> BaseModelT:
            """Call the model in structured output mode targeting the given Pydantic model async.

            Args:
                messages (list[Message]): The list of messages to send to the model.
                schema (type[BaseModelT]): The Pydantic model to use for the response.
                **kwargs: Additional keyword arguments to pass to the model.

            Returns:
                BaseModelT: The structured response from the model.

            """
            if schema.__name__ == "StepsOrError":
                return await self.aget_structured_response_instructor(messages, schema)
            langchain_messages = [msg.to_langchain() for msg in messages]
            structured_client = self._client.with_structured_output(
                schema, include_raw=True, **kwargs
            )
            raw_response = await structured_client.ainvoke(langchain_messages)
            if not isinstance(raw_response, dict):
                raise TypeError(f"Expected dict, got {type(raw_response).__name__}.")
            response = raw_response["parsed"]
            if isinstance(response, schema):
                return response
            return schema.model_validate(response)

        async def aget_structured_response_instructor(
            self,
            messages: list[Message],
            schema: type[BaseModelT],
        ) -> BaseModelT:
            """Get structured response using instructor asynchronously."""
            super()._log_llm_call(messages)
            instructor_messages = [map_message_to_instructor(msg) for msg in messages]
            return await self._acached_instructor_call(
                client=self._instructor_client_async,
                messages=instructor_messages,
                schema=schema,
                model=self.model_name,
                provider=self.provider.value,
            )


if validate_extras_dependencies("amazon", raise_error=False):
    import logging

    import boto3
    from langchain_aws import ChatBedrock

    def set_amazon_logging_level(level: int) -> None:
        """Set the logging level for boto3 client."""
        boto3.set_stream_logger(name="botocore.credentials", level=level)
        boto3.set_stream_logger(name="langchain_aws.llms.bedrock", level=level)

    class AmazonBedrockGenerativeModel(LangChainGenerativeModel):
        """Amazon Bedrock model implementation."""

        provider: LLMProvider = LLMProvider.AMAZON

        def __init__(
            self,
            *,
            model_id: str = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
            credentials_profile_name: str | None = None,
            aws_access_key_id: str | None = None,
            aws_secret_access_key: str | None = None,
            temperature: float | None = None,
            region_name: str | None = None,
            provider: str | None = None,
            **kwargs: Any,
        ) -> None:
            """Initialize with Amazon Bedrock client.

            Args:
                model_id: Name of the Amazon Bedrock model or the urn of the model.
                credentials_profile_name: Name of the AWS credentials profile to use
                  (loaded from ~/.aws/credentials), if not provided, both aws keys must be provided.
                aws_access_key_id: AWS access key ID is used, if credentials_profile_name
                 is not provided
                aws_secret_access_key: AWS secret access key is used, if aws_access_key_id
                 is provided.
                region_name: AWS region name, if not provided will be loaded
                 from the credentials profile.
                temperature: Temperature parameter for model sampling
                provider: Provider name if urn is provided in model_id
                 (e.g. "anthropic", "amazon", "meta"..etc)
                **kwargs: Additional keyword arguments to pass to ChatBedrock

            """
            set_amazon_logging_level(logging.WARNING)
            client = ChatBedrock(
                model=model_id,
                credentials_profile_name=credentials_profile_name,
                aws_access_key_id=SecretStr(aws_access_key_id) if aws_access_key_id else None,
                aws_secret_access_key=SecretStr(aws_secret_access_key)
                if aws_secret_access_key
                else None,
                region=region_name,
                provider=provider,
                temperature=temperature or 0,
                **kwargs,
            )
            super().__init__(client, model_id)
            bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=region_name,
                aws_access_key_id=SecretStr(aws_access_key_id) if aws_access_key_id else None,
                aws_secret_access_key=SecretStr(aws_secret_access_key)
                if aws_secret_access_key
                else None,
            )

            self._instructor_client = instructor.from_bedrock(bedrock_client)


if validate_extras_dependencies("google", raise_error=False):
    from google import genai
    from langchain_google_genai import ChatGoogleGenerativeAI

    from portia.gemini_langsmith_wrapper import wrap_gemini

    class GoogleGenAiGenerativeModel(LangChainGenerativeModel):
        """Google Generative AI (Gemini)model implementation."""

        provider: LLMProvider = LLMProvider.GOOGLE

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
            genai_client = genai.Client(api_key=api_key.get_secret_value())

            client = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=api_key,
                max_retries=max_retries,
                temperature=temperature or 0,
                **kwargs,
            )
            super().__init__(client, model_name)
            wrapped_gemini_client = wrap_gemini(genai_client)

            self._instructor_client = instructor.from_genai(
                client=wrapped_gemini_client,
                mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
            )
            self._instructor_client_async = instructor.from_genai(
                client=wrapped_gemini_client,
                mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
                use_async=True,
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
            instructor_messages = [map_message_to_instructor(message) for message in messages]
            return self._cached_instructor_call(
                client=client,
                messages=instructor_messages,
                schema=schema,
                model=self.model_name,
                provider=self.provider.value,
                max_retries=2,
            )

        async def aget_response(self, messages: list[Message]) -> Message:
            """Get response from Ollama model asynchronously."""
            langchain_messages = [msg.to_langchain() for msg in messages]
            response = await self._client.ainvoke(langchain_messages)
            return Message.from_langchain(response)

        async def aget_structured_response(
            self,
            messages: list[Message],
            schema: type[BaseModelT],
            **kwargs: Any,  # noqa: ARG002
        ) -> BaseModelT:
            """Get structured response from Ollama model asynchronously.

            Args:
                messages (list[Message]): The list of messages to send to the model.
                schema (type[BaseModelT]): The Pydantic model to use for the response.
                **kwargs: Additional keyword arguments to pass to the model.

            Returns:
                BaseModelT: The structured response from the model.

            """
            client = instructor.from_openai(
                AsyncOpenAI(
                    base_url=self.base_url,
                    api_key="ollama",  # required, but unused
                ),
                mode=instructor.Mode.JSON,
            )
            instructor_messages = [map_message_to_instructor(message) for message in messages]
            return await self._acached_instructor_call(
                client=client,
                messages=instructor_messages,
                schema=schema,
                model=self.model_name,
                provider=self.provider.value,
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
