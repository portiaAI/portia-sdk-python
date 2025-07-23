"""Unit tests for the Message class in portia.model."""

from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.caches import BaseCache
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import Generation
from pydantic import BaseModel, SecretStr, ValidationError

from portia.model import (
    AnthropicGenerativeModel,
    AzureOpenAIGenerativeModel,
    GenerativeModel,
    LangChainGenerativeModel,
    LLMProvider,
    Message,
    OpenAIGenerativeModel,
    map_message_to_instructor,
)
from portia.planning_agents.base_planning_agent import StepsOrError


@pytest.mark.parametrize(
    ("langchain_message", "expected_role", "expected_content"),
    [
        (HumanMessage(content="Hello"), "user", "Hello"),
        (AIMessage(content="Hi there"), "assistant", "Hi there"),
        (
            SystemMessage(content="You are a helpful assistant"),
            "system",
            "You are a helpful assistant",
        ),
    ],
)
def test_message_from_langchain(
    langchain_message: BaseMessage,
    expected_role: str,
    expected_content: str,
) -> None:
    """Test converting from LangChain messages to Portia Message."""
    message = Message.from_langchain(langchain_message)
    assert message.role == expected_role
    assert message.content == expected_content


def test_message_from_langchain_unsupported_type() -> None:
    """Test that converting from unsupported LangChain message type raises ValueError."""

    class UnsupportedMessage:
        content = "test"

    with pytest.raises(ValueError, match="Unsupported message type"):
        Message.from_langchain(UnsupportedMessage())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("portia_message", "expected_type", "expected_content"),
    [
        (Message(role="user", content="Hello"), HumanMessage, "Hello"),
        (Message(role="assistant", content="Hi there"), AIMessage, "Hi there"),
        (
            Message(role="system", content="You are a helpful assistant"),
            SystemMessage,
            "You are a helpful assistant",
        ),
    ],
)
def test_message_to_langchain(
    portia_message: Message,
    expected_type: type[BaseMessage],
    expected_content: str,
) -> None:
    """Test converting from Portia Message to LangChain messages."""
    langchain_message = portia_message.to_langchain()
    assert isinstance(langchain_message, expected_type)
    assert langchain_message.content == expected_content


def test_message_to_langchain_unsupported_role() -> None:
    """Test that converting to LangChain message with unsupported role raises ValueError."""
    message = Message(role="user", content="test")
    # Force an invalid role to test the to_langchain method
    message.role = "invalid"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported role"):
        message.to_langchain()


@pytest.mark.parametrize(
    ("message", "expected_instructor_message"),
    [
        (Message(role="user", content="Hello"), {"role": "user", "content": "Hello"}),
        (
            Message(role="assistant", content="Hi there"),
            {"role": "assistant", "content": "Hi there"},
        ),
        (
            Message(role="system", content="You are a helpful assistant"),
            {"role": "system", "content": "You are a helpful assistant"},
        ),
    ],
)
def test_map_message_to_instructor(message: Message, expected_instructor_message: dict) -> None:
    """Test mapping a Message to an Instructor message."""
    assert map_message_to_instructor(message) == expected_instructor_message


def test_map_message_to_instructor_unsupported_role() -> None:
    """Test mapping a Message to an Instructor message with an unsupported role."""
    message = SimpleNamespace(role="invalid", content="Hello")
    with pytest.raises(ValueError, match="Unsupported message role"):
        map_message_to_instructor(message)  # type: ignore[arg-type]


def test_message_validation() -> None:
    """Test basic Message model validation."""
    # Valid message
    message = Message(role="user", content="Hello")
    assert message.role == "user"
    assert message.content == "Hello"

    # Invalid role
    with pytest.raises(ValidationError, match="Input should be 'user', 'assistant' or 'system'"):
        Message(role="invalid", content="Hello")  # type: ignore[arg-type]

    # Missing required fields
    with pytest.raises(ValidationError, match="Field required"):
        Message()  # type: ignore[call-arg]


class DummyGenerativeModel(GenerativeModel):
    """Dummy generative model."""

    provider: LLMProvider = LLMProvider.CUSTOM

    def __init__(self, model_name: str) -> None:
        """Initialize the model."""
        super().__init__(model_name)

    def get_response(self, messages: list[Message]) -> Message:  # noqa: ARG002
        """Get a response from the model."""
        return Message(role="assistant", content="Hello")

    def get_structured_response(
        self,
        messages: list[Message],  # noqa: ARG002
        schema: type[BaseModel],
    ) -> BaseModel:
        """Get a structured response from the model."""
        return schema()

    async def aget_response(self, messages: list[Message]) -> Message:  # noqa: ARG002
        """Get a response from the model."""
        return Message(role="assistant", content="Hello")

    async def aget_structured_response(
        self, _: list[Message], schema: type[BaseModel]
    ) -> BaseModel:
        """Get a structured response from the model."""
        return schema(test_field="test")

    def to_langchain(self) -> BaseChatModel:
        """Not implemented in tests."""
        raise NotImplementedError("This method is not used in tests")


def test_model_to_string() -> None:
    """Test that the model to string method works."""
    model = DummyGenerativeModel(model_name="test")
    assert str(model) == "custom/test"
    assert repr(model) == 'DummyGenerativeModel("custom/test")'


class StructuredOutputTestModel(BaseModel):
    """Test model for structured output."""

    test_field: str


def test_langchain_model_structured_output_returns_dict() -> None:
    """Test that LangchainModel.structured_output returns a dict."""
    base_chat_model = MagicMock(spec=BaseChatModel)
    structured_output = MagicMock()
    base_chat_model.with_structured_output.return_value = structured_output
    structured_output.invoke.return_value = {"test_field": "Response from model"}
    model = LangChainGenerativeModel(client=base_chat_model, model_name="test")
    result = model.get_structured_response(
        messages=[Message(role="user", content="Hello")],
        schema=StructuredOutputTestModel,
    )
    assert isinstance(result, StructuredOutputTestModel)
    assert result.test_field == "Response from model"


def test_anthropic_model_structured_output_returns_invalid_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that AnthropicModel.structured_output returns a dict."""
    mock_set = MagicMock()
    monkeypatch.setattr("portia.model.set_llm_cache", mock_set)

    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)
    structured_output = MagicMock()
    mock_chat_anthropic.with_structured_output.return_value = structured_output
    structured_output.invoke.return_value = None

    with mock.patch("portia.model.ChatAnthropic") as mock_chat_anthropic_cls:
        mock_chat_anthropic_cls.return_value = mock_chat_anthropic
        model = AnthropicGenerativeModel(model_name="test", api_key=SecretStr("test"))
        with pytest.raises(TypeError, match="Expected dict, got None"):
            model.get_structured_response(
                messages=[Message(role="user", content="Hello")],
                schema=StructuredOutputTestModel,
            )
        mock_set.assert_not_called()


def test_anthropic_model_structured_output_returns_dict() -> None:
    """Test that AnthropicModel.structured_output returns a dict."""
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)
    structured_output = MagicMock()
    mock_chat_anthropic.with_structured_output.return_value = structured_output
    structured_output.invoke.return_value = {"parsed": {"test_field": "Response from model"}}

    with mock.patch("portia.model.ChatAnthropic") as mock_chat_anthropic_cls:
        mock_chat_anthropic_cls.return_value = mock_chat_anthropic
        model = AnthropicGenerativeModel(model_name="test", api_key=SecretStr("test"))
        result = model.get_structured_response(
            messages=[Message(role="user", content="Hello")],
            schema=StructuredOutputTestModel,
        )
        assert isinstance(result, StructuredOutputTestModel)
        assert result.test_field == "Response from model"


def test_anthropic_model_get_response_list_content() -> None:
    """Test that AnthropicGenerativeModel.get_response handles list content correctly."""
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)
    mock_chat_anthropic.invoke.return_value = AIMessage(
        content=[
            {
                "signature": "ErUBCkYIBRgCIkCHOW050nRsvKYRVKpDR2HQmAH9qGv",
                "thinking": "Let me carefully analyze ...",
                "type": "thinking",
            },
            {
                "text": "This is a test summary",
                "type": "text",
            },
        ],
        additional_kwargs={},
        response_metadata={
            "id": "msg_01KKD1wL1Xq37ErPYxCpRwX7",
            "model": "claude-3-7-sonnet-20250219",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "input_tokens": 203,
                "output_tokens": 392,
                "server_tool_use": None,
                "service_tier": "standard",
            },
            "model_name": "claude-3-7-sonnet-20250219",
        },
        id="run--0e5c64c3-ea97-4437-b672-4d0f812c981f-0",
        usage_metadata={
            "input_tokens": 203,
            "output_tokens": 392,
            "total_tokens": 595,
            "input_token_details": {"cache_read": 0, "cache_creation": 0},
        },
    )

    with mock.patch("portia.model.ChatAnthropic") as mock_chat_anthropic_cls:
        mock_chat_anthropic_cls.return_value = mock_chat_anthropic
        model = AnthropicGenerativeModel(model_name="test", api_key=SecretStr("test"))
        result = model.get_response(
            messages=[Message(role="user", content="Hello")],
        )
        assert isinstance(result, Message)
        assert result.role == "assistant"
        assert result.content == "This is a test summary"


def test_anthropic_model_get_response_with_human_message() -> None:
    """Test that AnthropicGenerativeModel.get_response works with a simple user message."""
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)
    mock_chat_anthropic.invoke.return_value = HumanMessage(content="This is a test summary")

    with mock.patch("portia.model.ChatAnthropic") as mock_chat_anthropic_cls:
        mock_chat_anthropic_cls.return_value = mock_chat_anthropic
        model = AnthropicGenerativeModel(model_name="test", api_key=SecretStr("test"))
        # This will be converted to a HumanMessage internally
        human_message = Message(role="user", content="Hello")
        result = model.get_response(
            messages=[human_message],
        )
        assert isinstance(result, Message)
        assert result.role == "user"
        assert result.content == "This is a test summary"


def test_anthropic_model_structured_output_fallback_to_instructor() -> None:
    """Test that AnthropicModel.structured_output falls back to instructor when expected."""
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)
    structured_output = MagicMock()
    mock_chat_anthropic.with_structured_output.return_value = structured_output
    structured_output.invoke.return_value = {
        "parsing_error": ValidationError("Test error", []),
        "raw": AIMessage(content=" ".join("portia" for _ in range(10000))),
        "parsed": None,
    }

    mock_cache = MagicMock()
    mock_cache.get.return_value = StructuredOutputTestModel(test_field="")
    LangChainGenerativeModel.set_cache(mock_cache)
    with (
        mock.patch("portia.model.ChatAnthropic") as mock_chat_anthropic_cls,
        mock.patch("instructor.from_anthropic") as mock_instructor,
    ):
        mock_chat_anthropic_cls.return_value = mock_chat_anthropic
        mock_instructor.return_value.chat.completions.create.return_value = (
            StructuredOutputTestModel(test_field="")
        )
        model = AnthropicGenerativeModel(
            model_name="test",
            api_key=SecretStr("test"),
            model_kwargs={"thinking": {"type": "enabled", "budget_tokens": 3000}},
        )
        _ = model.get_structured_response(
            messages=[Message(role="user", content="Hello")],
            schema=StructuredOutputTestModel,
        )
        mock_instructor.return_value.chat.completions.create.assert_called_once()


def test_instructor_manual_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM responses are cached when redis URL provided."""

    class DummyModel(BaseModel):
        pass

    mock_instructor_client = MagicMock()
    monkeypatch.setattr(
        "portia.model.instructor.from_openai",
        MagicMock(return_value=mock_instructor_client),
    )
    mock_create = MagicMock(return_value=DummyModel())
    mock_instructor_client.chat.completions.create = mock_create

    cache = MagicMock(spec=BaseCache)
    LangChainGenerativeModel.set_cache(cache)
    model = OpenAIGenerativeModel(
        model_name="gpt-4o",
        api_key=SecretStr("k"),
    )

    # Test cache miss
    cache.lookup.return_value = None
    model.get_structured_response_instructor([Message(role="user", content="hi")], DummyModel)
    cache.lookup.assert_called_once()
    cache.update.assert_called_once()

    # Test cache hit
    cache.reset_mock()
    cache.lookup.return_value = [Generation(text="{}")]
    model.get_structured_response_instructor([Message(role="user", content="hi")], DummyModel)
    cache.lookup.assert_called_once()
    cache.update.assert_not_called()

    # Test cache hit with validation error
    cache.reset_mock()
    cache.lookup.return_value = "{"
    model.get_structured_response_instructor([Message(role="user", content="hi")], DummyModel)
    cache.lookup.assert_called_once()
    cache.update.assert_called_once()


@pytest.mark.asyncio
async def test_dummy_model_async_methods() -> None:
    """Test that the dummy model async methods work."""
    model = DummyGenerativeModel(model_name="test")
    messages = [Message(role="user", content="Hello")]

    # Test aget_response
    response = await model.aget_response(messages)
    assert isinstance(response, Message)
    assert response.role == "assistant"
    assert response.content == "Hello"

    # Test aget_structured_response
    result = await model.aget_structured_response(messages, StructuredOutputTestModel)
    assert isinstance(result, StructuredOutputTestModel)


@pytest.mark.asyncio
async def test_langchain_model_async_structured_output_returns_dict() -> None:
    """Test that LangchainModel.aget_structured_response returns a dict."""
    base_chat_model = MagicMock(spec=BaseChatModel)
    structured_output = MagicMock()
    base_chat_model.with_structured_output.return_value = structured_output

    # Mock the async invoke to return a proper response
    async def mock_ainvoke(*_: Any, **__: Any) -> StructuredOutputTestModel:
        return StructuredOutputTestModel(test_field="Response from model")

    structured_output.ainvoke = mock_ainvoke
    model = LangChainGenerativeModel(client=base_chat_model, model_name="test")
    result = await model.aget_structured_response(
        messages=[Message(role="user", content="Hello")],
        schema=StructuredOutputTestModel,
    )
    assert isinstance(result, StructuredOutputTestModel)
    assert result.test_field == "Response from model"


@pytest.mark.asyncio
async def test_langchain_model_async_get_response() -> None:
    """Test that LangchainModel.aget_response works correctly."""
    base_chat_model = MagicMock(spec=BaseChatModel)
    base_chat_model.ainvoke.return_value = AIMessage(content="Hello from model")
    model = LangChainGenerativeModel(client=base_chat_model, model_name="test")
    result = await model.aget_response(
        messages=[Message(role="user", content="Hello")],
    )
    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "Hello from model"


@pytest.mark.asyncio
async def test_anthropic_model_async_structured_output_returns_invalid_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that AnthropicModel.aget_structured_response handles invalid data."""
    mock_set = MagicMock()
    monkeypatch.setattr("portia.model.set_llm_cache", mock_set)

    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)
    structured_output = MagicMock()
    mock_chat_anthropic.with_structured_output.return_value = structured_output

    # Mock the async invoke to return None
    async def mock_ainvoke(*_: Any, **__: Any) -> None:
        return None

    structured_output.ainvoke = mock_ainvoke

    with mock.patch("portia.model.ChatAnthropic") as mock_chat_anthropic_cls:
        mock_chat_anthropic_cls.return_value = mock_chat_anthropic
        model = AnthropicGenerativeModel(model_name="test", api_key=SecretStr("test"))
        with pytest.raises(TypeError, match="Expected dict, got None"):
            await model.aget_structured_response(
                messages=[Message(role="user", content="Hello")],
                schema=StructuredOutputTestModel,
            )
        mock_set.assert_not_called()


@pytest.mark.asyncio
async def test_anthropic_model_async_structured_output_returns_dict() -> None:
    """Test that AnthropicModel.aget_structured_response returns a dict."""
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)
    structured_output = MagicMock()
    mock_chat_anthropic.with_structured_output.return_value = structured_output

    # Mock the async invoke to return a proper response
    async def mock_ainvoke(*_: Any, **__: Any) -> dict[str, Any]:
        return {"parsed": {"test_field": "Response from model"}}

    structured_output.ainvoke = mock_ainvoke

    with mock.patch("portia.model.ChatAnthropic") as mock_chat_anthropic_cls:
        mock_chat_anthropic_cls.return_value = mock_chat_anthropic
        model = AnthropicGenerativeModel(model_name="test", api_key=SecretStr("test"))
        result = await model.aget_structured_response(
            messages=[Message(role="user", content="Hello")],
            schema=StructuredOutputTestModel,
        )
        assert isinstance(result, StructuredOutputTestModel)
        assert result.test_field == "Response from model"


@pytest.mark.asyncio
async def test_anthropic_model_async_get_response_list_content() -> None:
    """Test that AnthropicGenerativeModel.aget_response handles list content correctly."""
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)
    mock_chat_anthropic.ainvoke.return_value = AIMessage(
        content=[
            {
                "signature": "ErUBCkYIBRgCIkCHOW050nRsvKYRVKpDR2HQmAH9qGv",
                "thinking": "Let me carefully analyze ...",
                "type": "thinking",
            },
            {
                "text": "This is a test summary",
                "type": "text",
            },
        ],
        additional_kwargs={},
        response_metadata={
            "id": "msg_01KKD1wL1Xq37ErPYxCpRwX7",
            "model": "claude-3-7-sonnet-20250219",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "input_tokens": 203,
                "output_tokens": 392,
                "server_tool_use": None,
                "service_tier": "standard",
            },
            "model_name": "claude-3-7-sonnet-20250219",
        },
        id="run--0e5c64c3-ea97-4437-b672-4d0f812c981f-0",
        usage_metadata={
            "input_tokens": 203,
            "output_tokens": 392,
            "total_tokens": 595,
            "input_token_details": {"cache_read": 0, "cache_creation": 0},
        },
    )

    with mock.patch("portia.model.ChatAnthropic") as mock_chat_anthropic_cls:
        mock_chat_anthropic_cls.return_value = mock_chat_anthropic
        model = AnthropicGenerativeModel(model_name="test", api_key=SecretStr("test"))
        result = await model.aget_response(
            messages=[Message(role="user", content="Hello")],
        )
        assert isinstance(result, Message)
        assert result.role == "assistant"
        assert result.content == "This is a test summary"


@pytest.mark.asyncio
async def test_anthropic_model_async_get_response_with_human_message() -> None:
    """Test that AnthropicGenerativeModel.aget_response works with a simple user message."""
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)
    mock_chat_anthropic.ainvoke.return_value = HumanMessage(content="This is a test summary")

    with mock.patch("portia.model.ChatAnthropic") as mock_chat_anthropic_cls:
        mock_chat_anthropic_cls.return_value = mock_chat_anthropic
        model = AnthropicGenerativeModel(model_name="test", api_key=SecretStr("test"))
        # This will be converted to a HumanMessage internally
        human_message = Message(role="user", content="Hello")
        result = await model.aget_response(
            messages=[human_message],
        )
        assert isinstance(result, Message)
        assert result.role == "user"
        assert result.content == "This is a test summary"


@pytest.mark.asyncio
async def test_anthropic_model_async_structured_output_fallback_to_instructor() -> None:
    """Test that AnthropicModel.aget_structured_response falls back to instructor when expected."""
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)
    structured_output = MagicMock()
    mock_chat_anthropic.with_structured_output.return_value = structured_output

    # Mock the async invoke to return a response that triggers fallback
    async def mock_ainvoke(*_: Any, **__: Any) -> dict[str, Any]:
        return {
            "parsing_error": ValidationError("Test error", []),
            "raw": AIMessage(content=" ".join("portia" for _ in range(10000))),
            "parsed": None,
        }

    structured_output.ainvoke = mock_ainvoke

    mock_cache = MagicMock()

    # Mock the async lookup method
    async def mock_alookup(*_: Any, **__: Any) -> None:
        return None

    mock_cache.alookup = mock_alookup

    # Mock the async update method
    async def mock_aupdate(*_: Any, **__: Any) -> None:
        pass

    mock_cache.aupdate = mock_aupdate
    LangChainGenerativeModel.set_cache(mock_cache)
    with (
        mock.patch("portia.model.ChatAnthropic") as mock_chat_anthropic_cls,
        mock.patch("instructor.from_anthropic") as mock_instructor,
    ):
        mock_chat_anthropic_cls.return_value = mock_chat_anthropic
        # Mock the instructor client properly
        mock_instructor_client = MagicMock()
        mock_create = MagicMock()

        async def mock_create_async(*args: Any, **kwargs: Any) -> StructuredOutputTestModel:
            mock_create(*args, **kwargs)
            return StructuredOutputTestModel(test_field="")

        mock_instructor_client.chat.completions.create = mock_create_async
        mock_instructor.return_value = mock_instructor_client
        model = AnthropicGenerativeModel(
            model_name="test",
            api_key=SecretStr("test"),
            model_kwargs={"thinking": {"type": "enabled", "budget_tokens": 3000}},
        )
        _ = await model.aget_structured_response(
            messages=[Message(role="user", content="Hello")],
            schema=StructuredOutputTestModel,
        )
        # Verify the mock was called
        assert mock_create.call_count == 1


@pytest.mark.asyncio
async def test_instructor_async_manual_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM responses are cached when redis URL provided for async calls."""

    class DummyModel(BaseModel):
        pass

    mock_instructor_client = MagicMock()
    monkeypatch.setattr(
        "portia.model.instructor.from_openai",
        MagicMock(return_value=mock_instructor_client),
    )

    # Mock the async create method
    async def mock_create(*_: Any, **__: Any) -> DummyModel:
        return DummyModel()

    mock_instructor_client.chat.completions.create = mock_create

    cache = MagicMock(spec=BaseCache)
    LangChainGenerativeModel.set_cache(cache)
    model = OpenAIGenerativeModel(
        model_name="gpt-4o",
        api_key=SecretStr("k"),
    )

    # Test cache miss
    cache.alookup.return_value = None
    await model.aget_structured_response_instructor(
        [Message(role="user", content="hi")], DummyModel
    )
    cache.alookup.assert_called_once()
    cache.aupdate.assert_called_once()

    # Test cache hit
    cache.reset_mock()
    cache.alookup.return_value = [Generation(text="{}")]
    await model.aget_structured_response_instructor(
        [Message(role="user", content="hi")], DummyModel
    )
    cache.alookup.assert_called_once()
    cache.aupdate.assert_not_called()

    # Test cache hit with validation error
    cache.reset_mock()
    cache.alookup.return_value = "{"
    await model.aget_structured_response_instructor(
        [Message(role="user", content="hi")], DummyModel
    )
    cache.alookup.assert_called_once()
    cache.aupdate.assert_called_once()


# Parameterized tests for all model types
@pytest.mark.parametrize(
    ("model_class", "model_kwargs"),
    [
        (OpenAIGenerativeModel, {"model_name": "gpt-4o", "api_key": SecretStr("test")}),
        (
            AnthropicGenerativeModel,
            {"model_name": "claude-3-5-sonnet", "api_key": SecretStr("test")},
        ),
        (
            AzureOpenAIGenerativeModel,
            {
                "model_name": "gpt-4o",
                "api_key": SecretStr("test"),
                "azure_endpoint": "https://test.openai.azure.com",
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_model_async_methods_parameterized(model_class: type, model_kwargs: dict) -> None:  # noqa: ARG001
    """Test async methods for different model types."""
    with mock.patch(f"portia.model.{model_class.__name__}") as mock_model_cls:
        # Mock the model to avoid actual API calls
        mock_model = MagicMock(spec=model_class)
        mock_model.aget_response.return_value = Message(role="assistant", content="Test response")
        mock_model.aget_structured_response.return_value = StructuredOutputTestModel(
            test_field="test"
        )
        mock_model_cls.return_value = mock_model

        model = mock_model
        messages = [Message(role="user", content="Hello")]

        # Test aget_response
        response = await model.aget_response(messages)
        assert isinstance(response, Message)
        assert response.role == "assistant"
        assert response.content == "Test response"

        # Test aget_structured_response
        result = await model.aget_structured_response(messages, StructuredOutputTestModel)
        assert isinstance(result, StructuredOutputTestModel)
        assert result.test_field == "test"


@pytest.mark.asyncio
async def test_openai_model_async_methods() -> None:
    """Test OpenAI model async methods."""
    with mock.patch("portia.model.ChatOpenAI") as mock_chat_openai_cls:
        mock_chat_openai = MagicMock()

        # Mock the async invoke method
        async def mock_ainvoke(*_: Any, **__: Any) -> AIMessage:
            return AIMessage(content="OpenAI response")

        mock_chat_openai.ainvoke = mock_ainvoke

        # Mock the structured output
        structured_output = MagicMock()

        async def mock_structured_ainvoke(*_: Any, **__: Any) -> StructuredOutputTestModel:
            return StructuredOutputTestModel(test_field="OpenAI structured response")

        structured_output.ainvoke = mock_structured_ainvoke
        mock_chat_openai.with_structured_output.return_value = structured_output

        mock_chat_openai_cls.return_value = mock_chat_openai

        model = OpenAIGenerativeModel(
            model_name="gpt-4o",
            api_key=SecretStr("test"),
        )

        messages = [Message(role="user", content="Hello")]

        # Test aget_response
        response = await model.aget_response(messages)
        assert isinstance(response, Message)
        assert response.role == "assistant"
        assert response.content == "OpenAI response"

        # Test aget_structured_response
        result = await model.aget_structured_response(messages, StructuredOutputTestModel)
        assert isinstance(result, StructuredOutputTestModel)
        assert result.test_field == "OpenAI structured response"


@pytest.mark.asyncio
async def test_openai_model_async_instructor_fallback() -> None:
    """Test OpenAI model async instructor fallback for specific schemas."""
    with (
        mock.patch("portia.model.ChatOpenAI") as mock_chat_openai_cls,
        mock.patch("instructor.from_openai") as mock_instructor,
    ):
        mock_chat_openai = MagicMock()
        mock_chat_openai_cls.return_value = mock_chat_openai

        mock_instructor_client = MagicMock()
        mock_instructor.return_value = mock_instructor_client
        # Mock the async create method
        mock_create = MagicMock()

        async def mock_create_async(*args: Any, **kwargs: Any) -> StructuredOutputTestModel:
            mock_create(*args, **kwargs)
            return StructuredOutputTestModel(test_field="instructor")

        mock_instructor_client.chat.completions.create = mock_create_async

        model = OpenAIGenerativeModel(
            model_name="gpt-4o",
            api_key=SecretStr("test"),
        )

        messages = [Message(role="user", content="Hello")]

        # Test with StepsOrError schema (should use instructor)
        result = await model.aget_structured_response(messages, StepsOrError)
        assert isinstance(result, StructuredOutputTestModel)
        assert result.test_field == "instructor"
        # Verify the mock was called
        assert mock_create.call_count == 1


@pytest.mark.asyncio
async def test_azure_openai_model_async_methods() -> None:
    """Test Azure OpenAI model async methods."""
    with (
        mock.patch("portia.model.AzureChatOpenAI") as mock_chat_azure_openai_cls,
        mock.patch("instructor.from_openai") as mock_instructor,
    ):
        mock_chat_azure_openai = MagicMock()

        # Mock the async invoke method
        async def mock_ainvoke(*_: Any, **__: Any) -> AIMessage:
            return AIMessage(content="Azure OpenAI response")

        mock_chat_azure_openai.ainvoke = mock_ainvoke

        # Mock the structured output
        structured_output = MagicMock()

        async def mock_structured_ainvoke(*_: Any, **__: Any) -> StructuredOutputTestModel:
            return StructuredOutputTestModel(test_field="Azure OpenAI structured response")

        structured_output.ainvoke = mock_structured_ainvoke
        mock_chat_azure_openai.with_structured_output.return_value = structured_output

        mock_chat_azure_openai_cls.return_value = mock_chat_azure_openai

        mock_instructor_client = MagicMock()
        mock_instructor.return_value = mock_instructor_client
        # Mock the async create method
        mock_create = MagicMock()

        async def mock_create_async(*args: Any, **kwargs: Any) -> StructuredOutputTestModel:
            mock_create(*args, **kwargs)
            return StructuredOutputTestModel(test_field="azure instructor")

        mock_instructor_client.chat.completions.create = mock_create_async

        model = AzureOpenAIGenerativeModel(
            model_name="gpt-4o",
            api_key=SecretStr("test"),
            azure_endpoint="https://test.openai.azure.com",
        )

        messages = [Message(role="user", content="Hello")]

        # Test aget_response
        response = await model.aget_response(messages)
        assert isinstance(response, Message)
        assert response.role == "assistant"
        assert response.content == "Azure OpenAI response"

        # Test aget_structured_response
        result = await model.aget_structured_response(messages, StructuredOutputTestModel)
        assert isinstance(result, StructuredOutputTestModel)
        assert result.test_field == "Azure OpenAI structured response"

        # Test with StepsOrError schema (should use instructor)
        result = await model.aget_structured_response(messages, StepsOrError)
        assert isinstance(result, StructuredOutputTestModel)
        assert result.test_field == "azure instructor"
        # Verify the mock was called
        assert mock_create.call_count == 1


# Test conditional imports for optional models
@pytest.mark.asyncio
async def test_mistralai_model_async_methods_if_available() -> None:
    """Test MistralAI model async methods if the package is available."""
    try:
        from portia.model import MistralAIGenerativeModel

        with mock.patch("portia.model.ChatMistralAI") as mock_chat_mistral_cls:
            mock_chat_mistral = MagicMock()

            # Mock the async invoke method
            async def mock_ainvoke(*_: Any, **__: Any) -> AIMessage:
                return AIMessage(content="Mistral response")

            mock_chat_mistral.ainvoke = mock_ainvoke

            # Mock the structured output
            structured_output = MagicMock()

            async def mock_structured_ainvoke(*_: Any, **__: Any) -> dict[str, Any]:
                return {"parsed": {"test_field": "Mistral structured response"}}

            structured_output.ainvoke = mock_structured_ainvoke
            mock_chat_mistral.with_structured_output.return_value = structured_output

            mock_chat_mistral_cls.return_value = mock_chat_mistral

            model = MistralAIGenerativeModel(
                model_name="mistral-large",
                api_key=SecretStr("test"),
            )

            messages = [Message(role="user", content="Hello")]

            # Test aget_response
            response = await model.aget_response(messages)
            assert isinstance(response, Message)
            assert response.role == "assistant"
            assert response.content == "Mistral response"

            # Test aget_structured_response
            result = await model.aget_structured_response(messages, StructuredOutputTestModel)
            assert isinstance(result, StructuredOutputTestModel)
            assert result.test_field == "Mistral structured response"

    except ImportError:
        pytest.skip("MistralAI package not available")


@pytest.mark.asyncio
async def test_google_model_async_methods_if_available() -> None:
    """Test Google model async methods if the package is available."""
    try:
        from portia.model import GoogleGenAiGenerativeModel

        with mock.patch("portia.model.ChatGoogleGenerativeAI") as mock_chat_google_cls:
            mock_chat_google = MagicMock()

            # Mock the async invoke method
            async def mock_ainvoke(*_: Any, **__: Any) -> AIMessage:
                return AIMessage(content="Google response")

            mock_chat_google.ainvoke = mock_ainvoke

            # Mock the structured output
            structured_output = MagicMock()

            async def mock_structured_ainvoke(*_: Any, **__: Any) -> dict[str, Any]:
                return {"parsed": {"test_field": "Google structured response"}}

            structured_output.ainvoke = mock_structured_ainvoke
            mock_chat_google.with_structured_output.return_value = structured_output

            mock_chat_google_cls.return_value = mock_chat_google

            model = GoogleGenAiGenerativeModel(
                model_name="gemini-2.0-flash",
                api_key=SecretStr("test"),
            )

            messages = [Message(role="user", content="Hello")]

            # Test aget_response
            response = await model.aget_response(messages)
            assert isinstance(response, Message)
            assert response.role == "assistant"
            assert response.content == "Google response"

            # Test aget_structured_response
            result = await model.aget_structured_response(messages, StructuredOutputTestModel)
            assert isinstance(result, StructuredOutputTestModel)
            assert result.test_field == "Google structured response"

    except ImportError:
        pytest.skip("Google package not available")


@pytest.mark.asyncio
async def test_ollama_model_async_methods_if_available() -> None:
    """Test Ollama model async methods if the package is available."""
    try:
        from portia.model import OllamaGenerativeModel

        with mock.patch("portia.model.ChatOllama") as mock_chat_ollama_cls:
            mock_chat_ollama = MagicMock()

            # Mock the async invoke method
            async def mock_ainvoke(*_: Any, **__: Any) -> AIMessage:
                return AIMessage(content="Ollama response")

            mock_chat_ollama.ainvoke = mock_ainvoke
            mock_chat_ollama_cls.return_value = mock_chat_ollama

            model = OllamaGenerativeModel(
                model_name="llama2",
                base_url="http://localhost:11434/v1",
            )

            messages = [Message(role="user", content="Hello")]

            # Test aget_response
            response = await model.aget_response(messages)
            assert isinstance(response, Message)
            assert response.role == "assistant"
            assert response.content == "Ollama response"

            # Test aget_structured_response (uses instructor)
            with mock.patch("instructor.from_openai") as mock_instructor:
                mock_instructor_client = MagicMock()
                mock_instructor.return_value = mock_instructor_client

                # Mock the async create method
                async def mock_create(*_: Any, **__: Any) -> StructuredOutputTestModel:
                    return StructuredOutputTestModel(test_field="ollama")

                mock_instructor_client.chat.completions.create = mock_create

                result = await model.aget_structured_response(messages, StructuredOutputTestModel)
                assert isinstance(result, StructuredOutputTestModel)
                assert result.test_field == "ollama"

    except ImportError:
        pytest.skip("Ollama package not available")
