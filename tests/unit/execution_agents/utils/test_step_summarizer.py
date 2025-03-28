"""Test summarizer model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

if TYPE_CHECKING:
    from pydantic import BaseModel

from portia.execution_agents.base_execution_agent import Output
from portia.execution_agents.utils.step_summarizer import StepSummarizer
from tests.utils import get_mock_base_chat_model, get_test_llm_wrapper


def test_summarizer_model_normal_output() -> None:
    """Test the summarizer model with valid tool message."""
    summary = AIMessage(content="Short summary")
    mock_invoker = get_mock_base_chat_model(response=summary)
    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name="test_tool",
        artifact=Output(value="Tool output value"),
    )

    summarizer_model = StepSummarizer(
        llm=get_test_llm_wrapper(mock_invoker).to_langchain(),
    )
    result = summarizer_model.invoke({"messages": [tool_message]})

    assert mock_invoker.invoke.called
    messages: list[BaseMessage] = mock_invoker.invoke.call_args[0][0]
    assert messages
    assert "You are a highly skilled summarizer" in messages[0].content
    assert "Tool output content" in messages[1].content

    # Check that summaries were added to the artifact
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary == "Short summary"


def test_summarizer_model_non_tool_message() -> None:
    """Test the summarizer model with non-tool message should not invoke the LLM."""
    mock_invoker = get_mock_base_chat_model()
    ai_message = AIMessage(content="AI message content")

    summarizer_model = StepSummarizer(
        llm=get_test_llm_wrapper(mock_invoker).to_langchain(),
    )
    result = summarizer_model.invoke({"messages": [ai_message]})

    assert not mock_invoker.invoke.called
    assert result["messages"][0] == ai_message


def test_summarizer_model_no_messages() -> None:
    """Test the summarizer model with empty message list should not invoke the LLM."""
    mock_invoker = get_mock_base_chat_model()

    summarizer_model = StepSummarizer(
        llm=get_test_llm_wrapper(mock_invoker).to_langchain(),
    )
    result = summarizer_model.invoke({"messages": []})

    assert not mock_invoker.invoke.called
    assert result["messages"] == [None]


def test_summarizer_model_error_handling() -> None:
    """Test the summarizer model error handling."""

    class TestError(Exception):
        """Test error."""

    def mock_invoke(**_: Any) -> AIMessage | BaseModel:
        """Mock invoke that raises an error."""
        raise TestError("Test error")

    mock_invoker = get_mock_base_chat_model()
    mock_invoker.invoke = mock_invoke  # type: ignore  # noqa: PGH003

    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name="test_tool",
        artifact=Output(value="Tool output value"),
    )

    summarizer_model = StepSummarizer(
        llm=get_test_llm_wrapper(mock_invoker).to_langchain(),
    )
    result = summarizer_model.invoke({"messages": [tool_message]})

    # Should return original message without summaries when error occurs
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary is None
