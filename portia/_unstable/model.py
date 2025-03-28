"""LLM provider model classes for Portia Agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import instructor
from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, SecretStr

from portia.common import validate_extras_dependencies
from portia.planning_agents.base_planning_agent import StepsOrError

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
                {"role": "user", "content": message.content},
            )
        if isinstance(message, AIMessage):
            return cls.model_validate(
                {"role": "assistant", "content": message.content},
            )
        if isinstance(message, SystemMessage):
            return cls.model_validate(
                {"role": "system", "content": message.content},
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


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


class Model(ABC):
    """Base class for all Model clients."""

    @abstractmethod
    def get_response(self, messages: list[Message]) -> Message:
        """Given a list of messages, call the model and return its response as a new message.

        Args:
            messages (list[Message]): The list of messages to send to the model.

        Returns:
            Message: The response from the model.

        """
        ...

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


class LangChainModel(Model):
    """Base class for LangChain-based models."""

    def __init__(self, client: BaseChatModel) -> None:
        """Initialize with LangChain client.

        Args:
            client: LangChain chat model instance

        """
        self._client = client

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

    def _map_message_to_instructor(self, message: Message) -> ChatCompletionMessageParam:
        """Type-safe mapping of Message to ChatCompletionMessageParam."""
        match message:
            case Message(role="user", content=content):
                return {"role": "user", "content": content}
            case Message(role="assistant", content=content):
                return {"role": "assistant", "content": content}
            case Message(role="system", content=content):
                return {"role": "system", "content": content}
            case _:
                raise ValueError(f"Unsupported message role: {message.role}")


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


class OpenAIModel(LangChainModel):
    """OpenAI model implementation."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_name: str,
        api_key: SecretStr,
        seed: int = 343,
        max_retries: int = 3,
        temperature: float = 0,
        disabled_params: dict[str, None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with OpenAI client.

        Args:
            model_name: OpenAI model to use
            api_key: API key for OpenAI
            seed: Random seed for model generation
            max_retries: Maximum number of retries
            temperature: Temperature parameter (defaults to 1 for O_3_MINI, 0 otherwise)
            disabled_params: Parameters to disable in the client
            **kwargs: Additional keyword arguments to pass to ChatOpenAI

        """
        if disabled_params is None:
            disabled_params = {"parallel_tool_calls": None}

        client = ChatOpenAI(
            name=model_name,
            model=model_name,
            seed=seed,
            api_key=api_key,
            max_retries=max_retries,
            temperature=temperature,
            disabled_params=disabled_params,
            **kwargs,
        )
        super().__init__(client)
        self._api_key = api_key
        self._model_name = model_name
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
        if schema == StepsOrError:
            return self.get_structured_response_instructor(messages, schema)
        return super().get_structured_response(
            messages, schema, method="function_calling", **kwargs,
        )

    def get_structured_response_instructor(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
    ) -> BaseModelT:
        """Get structured response using instructor."""
        instructor_client = instructor.from_openai(
            client=OpenAI(api_key=self._api_key.get_secret_value()),
            mode=instructor.Mode.JSON,
        )
        instructor_messages = [map_message_to_instructor(msg) for msg in messages]
        return instructor_client.chat.completions.create(
            response_model=schema,
            messages=instructor_messages,
            model=self._model_name,
            seed=self._seed,
        )


class AzureOpenAIModel(LangChainModel):
    """Azure OpenAI model implementation."""

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
        disabled_params: dict[str, None] | None = None,
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
            disabled_params: Parameters to disable in the client
            **kwargs: Additional keyword arguments to pass to AzureChatOpenAI

        """
        if disabled_params is None:
            disabled_params = {"parallel_tool_calls": None}

        client = AzureChatOpenAI(
            name=model_name,
            model=model_name,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            seed=seed,
            api_key=api_key,
            max_retries=max_retries,
            temperature=temperature,
            disabled_params=disabled_params,
            **kwargs,
        )
        super().__init__(client)
        self._api_key = api_key
        self._model_name = model_name
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version
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
        if schema == StepsOrError:
            return self.get_structured_response_instructor(messages, schema)
        return super().get_structured_response(
            messages, schema, method="function_calling", **kwargs,
        )

    def get_structured_response_instructor(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
    ) -> BaseModelT:
        """Get structured response using instructor."""
        instructor_client = instructor.from_openai(
            client=AzureOpenAI(
                api_key=self._api_key.get_secret_value(),
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            ),
            mode=instructor.Mode.JSON,
        )
        instructor_messages = [map_message_to_instructor(msg) for msg in messages]
        return instructor_client.chat.completions.create(
            response_model=schema,
            messages=instructor_messages,
            model=self._model_name,
            seed=self._seed,
        )


class AnthropicModel(LangChainModel):
    """Anthropic model implementation."""

    def __init__(
        self,
        *,
        model_name: str = "claude-3-5-sonnet-latest",
        api_key: SecretStr,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize with Anthropic client.

        Args:
            model_name: Name of the Anthropic model
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            api_key: API key for Anthropic
            **kwargs: Additional keyword arguments to pass to ChatAnthropic

        """
        client = ChatAnthropic(
            model_name=model_name,
            timeout=timeout,
            max_retries=max_retries,
            api_key=api_key,
            **kwargs,
        )
        super().__init__(client)
        self._api_key = api_key
        self._model_name = model_name

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
        if schema == StepsOrError:
            return self.get_structured_response_instructor(messages, schema)
        return super().get_structured_response(messages, schema, **kwargs)

    def get_structured_response_instructor(
        self,
        messages: list[Message],
        schema: type[BaseModelT],
    ) -> BaseModelT:
        """Get structured response using instructor."""
        instructor_client = instructor.from_anthropic(
            client=Anthropic(api_key=self._api_key.get_secret_value()),
            mode=instructor.Mode.ANTHROPIC_JSON,
        )
        instructor_messages = [map_message_to_instructor(msg) for msg in messages]
        return instructor_client.chat.completions.create(
            model=self._model_name,
            response_model=schema,
            messages=instructor_messages,
            max_tokens=2048,
        )


if validate_extras_dependencies("mistral", raise_error=False):
    from mistralai import Mistral

    class MistralAIModel(LangChainModel):
        """MistralAI model implementation."""

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
            super().__init__(client)
            self._api_key = api_key
            self._model_name = model_name

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
            if schema == StepsOrError:
                return self.get_structured_response_instructor(messages, schema)
            return super().get_structured_response(
                messages, schema, method="function_calling", **kwargs,
            )

        def get_structured_response_instructor(
            self,
            messages: list[Message],
            schema: type[BaseModelT],
        ) -> BaseModelT:
            """Get structured response using instructor."""
            instructor_client = instructor.from_mistral(
                client=Mistral(api_key=self._api_key.get_secret_value()),
                use_async=False,
            )
            instructor_messages = [map_message_to_instructor(msg) for msg in messages]
            return instructor_client.chat.completions.create(
                model=self._model_name,
                response_model=schema,
                messages=instructor_messages,
            )


if validate_extras_dependencies("google", raise_error=False):
    import google.generativeai as genai

    class GoogleGenerativeAIModel(LangChainModel):
        """Google Generative AI (Gemini)model implementation."""

        def __init__(
            self,
            *,
            model_name: str = "gemini-2.0-flash",
            api_key: SecretStr,
            max_retries: int = 3,
            **kwargs: Any,
        ) -> None:
            """Initialize with Google Generative AI client.

            Args:
                model_name: Name of the Google Generative AI model
                api_key: API key for Google Generative AI
                max_retries: Maximum number of retries
                **kwargs: Additional keyword arguments to pass to ChatGoogleGenerativeAI

            """
            client = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=api_key,
                max_retries=max_retries,
                **kwargs,
            )
            self._api_key = api_key
            self._model_name = model_name
            super().__init__(client)

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
            # Configure genai with the api key
            genai.configure(api_key=self._api_key.get_secret_value())  # pyright: ignore[reportPrivateImportUsage]

            # Create instructor client
            instructor_client = instructor.from_gemini(
                client=genai.GenerativeModel(model_name=self._model_name),  # pyright: ignore[reportPrivateImportUsage]
                mode=instructor.Mode.GEMINI_JSON,
                use_async=False,
            )

            # Convert messages to format expected by instructor
            instructor_messages = [map_message_to_instructor(msg) for msg in messages]

            return instructor_client.messages.create(
                messages=instructor_messages,
                response_model=schema,
            )
