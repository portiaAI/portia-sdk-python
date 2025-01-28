"""Test summarizer model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    import pytest
    from pydantic import BaseModel

from portia.agents.base_agent import Output
from portia.agents.utils.step_summarizer import LLMSummarizer
from portia.llm_wrapper import LLMWrapper
from tests.utils import get_test_config


class MockInvoker:
    """Mock invoker."""

    called: bool
    prompt: list[BaseMessage]
    response: AIMessage | BaseModel | None

    def __init__(self, response: AIMessage | BaseModel | None = None) -> None:
        """Init worker."""
        self.called = False
        self.prompt = []
        self.response = response
        self.output_format = None

    def invoke(
        self,
        prompt: list[BaseMessage],
        **_: Any,  # noqa: ANN401
    ) -> AIMessage | BaseModel:
        """Mock run for invoking the chain."""
        self.called = True
        self.prompt = prompt
        if self.response:
            return self.response
        return AIMessage(content="invoked")

    def with_structured_output(self, output_format: type[BaseModel]) -> MockInvoker:
        """Model wrapper for structured output."""
        self.output_format = output_format
        return self


def test_summarizer_model_normal_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the summarizer model with valid tool message."""
    summary = AIMessage(content="Short summary")
    mock_invoker = MockInvoker(response=summary)
    monkeypatch.setattr(ChatOpenAI, "invoke", mock_invoker.invoke)
    monkeypatch.setattr(ChatOpenAI, "with_structured_output", mock_invoker.with_structured_output)

    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name="test_tool",
        artifact=Output(value="Tool output value"),
    )

    summarizer_model = LLMSummarizer(
        llm=LLMWrapper(get_test_config()).to_langchain(),
    )
    result = summarizer_model.invoke({"messages": [tool_message]})

    assert mock_invoker.called
    messages: list[BaseMessage] = mock_invoker.prompt
    assert messages
    assert "You are a highly skilled summarizer" in messages[0].content
    assert "Tool output content" in messages[1].content

    # Check that summaries were added to the artifact
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary == "Short summary"


def test_summarizer_model_non_tool_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the summarizer model with non-tool message should not invoke the LLM."""
    mock_invoker = MockInvoker()
    monkeypatch.setattr(ChatOpenAI, "invoke", mock_invoker.invoke)
    monkeypatch.setattr(ChatOpenAI, "with_structured_output", mock_invoker.with_structured_output)

    ai_message = AIMessage(content="AI message content")

    summarizer_model = LLMSummarizer(
        llm=LLMWrapper(get_test_config()).to_langchain(),
    )
    result = summarizer_model.invoke({"messages": [ai_message]})

    assert not mock_invoker.called
    assert result["messages"][0] == ai_message


def test_summarizer_model_no_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the summarizer model with empty message list should not invoke the LLM."""
    mock_invoker = MockInvoker()
    monkeypatch.setattr(ChatOpenAI, "invoke", mock_invoker.invoke)
    monkeypatch.setattr(ChatOpenAI, "with_structured_output", mock_invoker.with_structured_output)

    summarizer_model = LLMSummarizer(
        llm=LLMWrapper(get_test_config()).to_langchain(),
    )
    result = summarizer_model.invoke({"messages": []})

    assert not mock_invoker.called
    assert result["messages"] == [None]


def test_summarizer_model_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the summarizer model error handling."""
    class TestError(Exception):
        """Test error."""

    def mock_invoke(**_: Any) -> AIMessage | BaseModel:  # noqa: ANN401
        """Mock invoke that raises an error."""
        raise TestError("Test error")

    mock_invoker = MockInvoker()
    mock_invoker.invoke = mock_invoke  # type: ignore  # noqa: PGH003
    monkeypatch.setattr(ChatOpenAI, "invoke", mock_invoker.invoke)
    monkeypatch.setattr(ChatOpenAI, "with_structured_output", mock_invoker.with_structured_output)

    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name="test_tool",
        artifact=Output(value="Tool output value"),
    )

    summarizer_model = LLMSummarizer(
        llm=LLMWrapper(get_test_config()).to_langchain(),
    )
    result = summarizer_model.invoke({"messages": [tool_message]})

    # Should return original message without summaries when error occurs
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary is None
