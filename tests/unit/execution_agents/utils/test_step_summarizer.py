"""Test summarizer model."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from portia.config import FEATURE_FLAG_AGENT_MEMORY_ENABLED
from portia.execution_agents.output import LocalOutput
from portia.execution_agents.utils.step_summarizer import StepSummarizer
from portia.plan import Step
from tests.utils import (
    AdditionTool,
    get_mock_langchain_generative_model,
    get_test_config,
)


def test_summarizer_model_normal_output() -> None:
    """Test the summarizer model with valid tool message."""
    summary = AIMessage(content="Short summary")
    tool = AdditionTool()
    mock_model = get_mock_langchain_generative_model(response=summary)
    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name=tool.name,
        artifact=LocalOutput(value="Tool output value"),
    )

    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=tool,
        step=Step(task="Test task", output="$output"),
    )
    base_chat_model = mock_model.to_langchain()
    result = summarizer_model.invoke({"messages": [tool_message]})

    assert base_chat_model.invoke.called  # type: ignore[reportFunctionMemberAccess]
    messages = base_chat_model.invoke.call_args[0][0]  # type: ignore[reportFunctionMemberAccess]
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
    ai_message = AIMessage(content="AI message content")

    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=AdditionTool(),
        step=Step(task="Test task", output="$output"),
    )
    result = summarizer_model.invoke({"messages": [ai_message]})

    assert not mock_model.to_langchain().invoke.called  # type: ignore[reportFunctionMemberAccess]
    assert result["messages"][0] == ai_message


def test_summarizer_model_no_messages() -> None:
    """Test the summarizer model with empty message list should not invoke the LLM."""
    mock_model = get_mock_langchain_generative_model()

    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=AdditionTool(),
        step=Step(task="Test task", output="$output"),
    )
    result = summarizer_model.invoke({"messages": []})

    assert not mock_model.to_langchain().invoke.called  # type: ignore[reportFunctionMemberAccess]
    assert result["messages"] == [None]


def test_summarizer_model_large_output() -> None:
    """Test the summarizer model with large output."""
    summary = AIMessage(content="Short summary")
    mock_model = get_mock_langchain_generative_model(response=summary)
    tool_message = ToolMessage(
        content="Test " * 1000,
        tool_call_id="123",
        name="test_tool",
        artifact=LocalOutput(value="Test " * 1000),
    )

    summarizer_model = StepSummarizer(
        # Set a low threshold so the above output is considered large
        config=get_test_config(
            large_output_threshold_tokens=100,
            feature_flags={
                FEATURE_FLAG_AGENT_MEMORY_ENABLED: True,
            },
        ),
        model=mock_model,
        tool=AdditionTool(),
        step=Step(task="Test task", output="$output"),
    )
    base_chat_model = mock_model.to_langchain()
    result = summarizer_model.invoke({"messages": [tool_message]})

    assert base_chat_model.invoke.called  # type: ignore[reportFunctionMemberAccess]
    messages = base_chat_model.invoke.call_args[0][0]  # type: ignore[reportFunctionMemberAccess]
    assert messages
    assert "You are a highly skilled summarizer" in messages[0].content
    assert "This is a large value" in messages[1].content
    # Check that the content has been truncated
    assert messages[1].content.count("Test") < 1000

    # Check that summaries were added to the artifact
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary == "Short summary"


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
        artifact=LocalOutput(value="Tool output value"),
    )

    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=AdditionTool(),
        step=Step(task="Test task", output="$output"),
    )
    result = summarizer_model.invoke({"messages": [tool_message]})

    # Should return original message without summaries when error occurs
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary is None
