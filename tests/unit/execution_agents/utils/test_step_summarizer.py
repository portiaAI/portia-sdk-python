"""Test summarizer model."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from portia.execution_agents.base_execution_agent import Output
from portia.execution_agents.utils.step_summarizer import StepSummarizer
from portia.model import Message
from portia.plan import Step
from tests.utils import AdditionTool, get_mock_langchain_generative_model


def test_summarizer_model_normal_output() -> None:
    """Test the summarizer model with valid tool message."""
    summary = AIMessage(content="Short summary")
    tool = AdditionTool()
    mock_model = get_mock_langchain_generative_model(response=summary)
    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name=tool.name,
        artifact=Output(value="Tool output value"),
    )

    summarizer_model = StepSummarizer(
        model=mock_model,
        tool=tool,
        step=Step(task="Test task", output="$output"),
    )
    base_chat_model = mock_model.to_langchain()
    result = summarizer_model.invoke({"messages": [tool_message]})

    assert base_chat_model.invoke.called
    messages: list[Message] = base_chat_model.invoke.call_args[0][0]
    assert messages
    assert "You are a highly skilled summarizer" in messages[0].content
    assert "Tool output content" in messages[1].content

    # Check that summaries were added to the artifact
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary == "Short summary"


def test_summarizer_model_non_tool_message() -> None:
    """Test the summarizer model with non-tool message should not invoke the LLM."""
    mock_model = get_mock_langchain_generative_model()
    ai_message = Message(role="assistant", content="AI message content")

    summarizer_model = StepSummarizer(
        model=mock_model,
        tool=AdditionTool(),
        step=Step(task="Test task", output="$output"),
    )
    result = summarizer_model.invoke({"messages": [ai_message]})

    assert not mock_model.to_langchain().invoke.called
    assert result["messages"][0] == ai_message


def test_summarizer_model_no_messages() -> None:
    """Test the summarizer model with empty message list should not invoke the LLM."""
    mock_model = get_mock_langchain_generative_model()

    summarizer_model = StepSummarizer(
        model=mock_model,
        tool=AdditionTool(),
        step=Step(task="Test task", output="$output"),
    )
    result = summarizer_model.invoke({"messages": []})

    assert not mock_model.to_langchain().invoke.called
    assert result["messages"] == [None]


def test_summarizer_model_error_handling() -> None:
    """Test the summarizer model error handling."""

    class TestError(Exception):
        """Test error."""

    mock_model = get_mock_langchain_generative_model()
    mock_model.to_langchain().invoke.side_effect = TestError("Test error")  # type: ignore[reportFunctionMemberAccess]

    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name="test_tool",
        artifact=Output(value="Tool output value"),
    )

    summarizer_model = StepSummarizer(
        model=mock_model,
        tool=AdditionTool(),
        step=Step(task="Test task", output="$output"),
    )
    result = summarizer_model.invoke({"messages": [tool_message]})

    # Should return original message without summaries when error occurs
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary is None
