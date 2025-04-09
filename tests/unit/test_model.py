"""Unit tests for the Message class in portia.model."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from portia.model import (
    GenerativeModel,
    LangChainGenerativeModel,
    LLMProvider,
    Message,
    map_message_to_instructor,
)


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
