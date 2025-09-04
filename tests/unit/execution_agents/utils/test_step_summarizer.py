"""Test summarizer model."""

from __future__ import annotations

from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from portia.execution_agents.output import LocalDataValue
from portia.execution_agents.utils.step_summarizer import StepSummarizer
from portia.plan import Step
from tests.utils import (
    AdditionTool,
    get_mock_generative_model,
    get_test_config,
)


class TestError(Exception):
    """Test exception for step summarizer tests."""


def test_summarizer_model_normal_output() -> None:
    """Test the summarizer model with valid tool message."""
    summary = AIMessage(content="Short summary")
    tool = AdditionTool()
    mock_model = get_mock_generative_model(response=summary)
    ai_message = AIMessage(content="", tool_calls=[{"id": "123", "name": tool.name, "args": {}}])
    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name=tool.name,
        artifact=LocalDataValue(value="Tool output value"),
    )

    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=tool,
        step=Step(task="Test task", output="$output"),
    )
    base_chat_model = mock_model.to_langchain()
    result = summarizer_model.invoke({"messages": [ai_message, tool_message]})

    assert base_chat_model.invoke.called  # type: ignore[reportFunctionMemberAccess]
    messages = base_chat_model.invoke.call_args[0][0]  # type: ignore[reportFunctionMemberAccess]
    assert messages
    assert "You are a highly skilled summarizer" in messages[0].content
    assert "Tool output content" in messages[1].content

    # Check that summaries were added to the artifact
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary == "Short summary"


@pytest.mark.asyncio
async def test_summarizer_model_normal_output_async() -> None:
    """Test the async summarizer model with valid tool message."""
    summary = AIMessage(content="Short summary")
    tool = AdditionTool()
    mock_model = get_mock_generative_model(response=summary)
    ai_message = AIMessage(content="", tool_calls=[{"id": "123", "name": tool.name, "args": {}}])
    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name=tool.name,
        artifact=LocalDataValue(value="Tool output value"),
    )

    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=tool,
        step=Step(task="Test task", output="$output"),
    )
    base_chat_model = mock_model.to_langchain()
    result = await summarizer_model.ainvoke({"messages": [ai_message, tool_message]})

    assert base_chat_model.ainvoke.called  # type: ignore[reportFunctionMemberAccess]
    messages = base_chat_model.ainvoke.call_args[0][0]  # type: ignore[reportFunctionMemberAccess]
    assert messages
    assert "You are a highly skilled summarizer" in messages[0].content
    assert "Tool output content" in messages[1].content

    # Check that summaries were added to the artifact
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary == "Short summary"


def test_summarizer_model_non_tool_message() -> None:
    """Test the summarizer model with non-tool message should not invoke the LLM."""
    mock_model = get_mock_generative_model()
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


@pytest.mark.asyncio
async def test_summarizer_model_non_tool_message_async() -> None:
    """Test the async summarizer model with non-tool message should not invoke the LLM."""
    mock_model = get_mock_generative_model()
    ai_message = AIMessage(content="AI message content")

    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=AdditionTool(),
        step=Step(task="Test task", output="$output"),
    )
    result = await summarizer_model.ainvoke({"messages": [ai_message]})

    assert not mock_model.to_langchain().ainvoke.called  # type: ignore[reportFunctionMemberAccess]
    assert result["messages"][0] == ai_message


def test_summarizer_model_no_messages() -> None:
    """Test the summarizer model with empty message list should not invoke the LLM."""
    mock_model = get_mock_generative_model()

    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=AdditionTool(),
        step=Step(task="Test task", output="$output"),
    )
    result = summarizer_model.invoke({"messages": []})

    assert not mock_model.to_langchain().invoke.called  # type: ignore[reportFunctionMemberAccess]
    assert result["messages"] == [None]


@pytest.mark.asyncio
async def test_summarizer_model_no_messages_async() -> None:
    """Test the async summarizer model with empty message list should not invoke the LLM."""
    mock_model = get_mock_generative_model()

    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=AdditionTool(),
        step=Step(task="Test task", output="$output"),
    )
    result = await summarizer_model.ainvoke({"messages": []})

    assert not mock_model.to_langchain().ainvoke.called  # type: ignore[reportFunctionMemberAccess]
    assert result["messages"] == [None]


def test_summarizer_model_large_output() -> None:
    """Test the summarizer model with large output."""
    summary = AIMessage(content="Short summary")
    mock_model = get_mock_generative_model(response=summary)
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": "123", "name": "test_tool", "args": {}}],
    )
    tool_message = ToolMessage(
        content="Test " * 1000,
        tool_call_id="123",
        name="test_tool",
        artifact=LocalDataValue(value="Test " * 1000),
    )

    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=AdditionTool(),
        step=Step(task="Test task", output="$output"),
    )
    base_chat_model = mock_model.to_langchain()
    with mock.patch(
        "portia.execution_agents.utils.step_summarizer.exceeds_context_threshold"
    ) as mock_threshold:
        mock_threshold.return_value = True
        result = summarizer_model.invoke({"messages": [ai_message, tool_message]})

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


@pytest.mark.asyncio
async def test_summarizer_model_large_output_async() -> None:
    """Test the async summarizer model with large output."""
    summary = AIMessage(content="Short summary")
    mock_model = get_mock_generative_model(response=summary)
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": "123", "name": "test_tool", "args": {}}],
    )
    tool_message = ToolMessage(
        content="Test " * 1000,
        tool_call_id="123",
        name="test_tool",
        artifact=LocalDataValue(value="Test " * 1000),
    )

    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=AdditionTool(),
        step=Step(task="Test task", output="$output"),
    )
    base_chat_model = mock_model.to_langchain()
    with mock.patch(
        "portia.execution_agents.utils.step_summarizer.exceeds_context_threshold"
    ) as mock_threshold:
        mock_threshold.return_value = True
        result = await summarizer_model.ainvoke({"messages": [ai_message, tool_message]})

    assert base_chat_model.ainvoke.called  # type: ignore[reportFunctionMemberAccess]
    messages = base_chat_model.ainvoke.call_args[0][0]  # type: ignore[reportFunctionMemberAccess]
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

    mock_model = get_mock_generative_model()
    mock_model.to_langchain().invoke.side_effect = TestError("Test error")  # type: ignore[reportFunctionMemberAccess]

    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name="test_tool",
        artifact=LocalDataValue(value="Tool output value"),
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


@pytest.mark.asyncio
async def test_summarizer_model_error_handling_async() -> None:
    """Test the async summarizer model error handling."""

    class TestError(Exception):
        """Test error."""

    mock_model = get_mock_generative_model()
    mock_model.to_langchain().ainvoke.side_effect = TestError("Test error")  # type: ignore[reportFunctionMemberAccess]

    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name="test_tool",
        artifact=LocalDataValue(value="Tool output value"),
    )

    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=AdditionTool(),
        step=Step(task="Test task", output="$output"),
    )
    result = await summarizer_model.ainvoke({"messages": [tool_message]})

    # Should return original message without summaries when error occurs
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary is None


def test_summarizer_model_structured_output_schema() -> None:
    """Test the summarizer model with structured output schema."""
    tool = AdditionTool()

    class AdditionOutput(BaseModel):
        result: int
        so_summary: str

    tool.structured_output_schema = AdditionOutput
    mock_model = MagicMock()
    output = AdditionOutput(result=3, so_summary="Short summary")
    mock_model.get_structured_response.return_value = output
    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name=tool.name,
        artifact=LocalDataValue(value=3),
    )
    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=tool,
        step=Step(task="Test task", output="$output"),
    )
    result = summarizer_model.invoke({"messages": [tool_message]})

    # Check that summaries were added to the artifact
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary == output.so_summary


@pytest.mark.asyncio
async def test_summarizer_model_structured_output_schema_async() -> None:
    """Test the async summarizer model with structured output schema."""
    tool = AdditionTool()

    class AdditionOutput(BaseModel):
        result: int
        so_summary: str

    tool.structured_output_schema = AdditionOutput
    mock_model = MagicMock()
    output = AdditionOutput(result=3, so_summary="Short summary")

    # Create an async mock that returns the output
    async def mock_aget_structured_response(*_args: Any, **_kwargs: Any) -> AdditionOutput:
        return output

    mock_model.aget_structured_response = mock_aget_structured_response
    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name=tool.name,
        artifact=LocalDataValue(value=3),
    )
    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=tool,
        step=Step(task="Test task", output="$output"),
    )
    result = await summarizer_model.ainvoke({"messages": [tool_message]})

    # Check that summaries were added to the artifact
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary == output.so_summary


def test_summarizer_model_structured_output_schema_error_fallback() -> None:
    """Test the summarizer model with structured output schema."""
    tool = AdditionTool()

    mock_model = MagicMock()

    tool.structured_output_schema = BaseModel
    mock_model.get_structured_response.side_effect = Exception("Test error")
    summary = AIMessage(content="Short Summary")
    mock_model.get_response.return_value = summary

    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name=tool.name,
        artifact=LocalDataValue(value=3),
    )
    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=tool,
        step=Step(task="Test task", output="$output"),
    )
    result = summarizer_model.invoke({"messages": [tool_message]})

    # Check that the tool message is returned unchanged
    output_message = result["messages"][0]
    assert output_message == tool_message


@pytest.mark.asyncio
async def test_summarizer_model_structured_output_schema_error_fallback_async() -> None:
    """Test the async summarizer model with structured output schema."""
    tool = AdditionTool()

    mock_model = MagicMock()

    tool.structured_output_schema = BaseModel

    # Create async mocks
    async def mock_aget_structured_response(*_args: Any, **_kwargs: Any) -> None:
        raise TestError("Test error")

    async def mock_aget_response(*_args: Any, **_kwargs: Any) -> AIMessage:
        return AIMessage(content="Short Summary")

    mock_model.aget_structured_response = mock_aget_structured_response
    mock_model.aget_response = mock_aget_response

    tool_message = ToolMessage(
        content="Tool output content",
        tool_call_id="123",
        name=tool.name,
        artifact=LocalDataValue(value=3),
    )
    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=tool,
        step=Step(task="Test task", output="$output"),
    )
    result = await summarizer_model.ainvoke({"messages": [tool_message]})

    # Check that the tool message is returned unchanged
    output_message = result["messages"][0]
    assert output_message == tool_message


def test_summarizer_model_multiple_tool_calls() -> None:
    """Test the summarizer model with multiple tool calls."""
    summary = AIMessage(content="Short summary")
    tool = AdditionTool()
    mock_model = get_mock_generative_model(response=summary)

    tool_call_1_id = "123"
    tool_call_2_id = "456"

    tool_call_1_args = {"a": 1, "b": 2}
    tool_call_2_args = {"a": 3, "b": 4}

    ai_message = AIMessage(
        content="",
        tool_calls=[
            {"id": tool_call_1_id, "name": tool.name, "args": tool_call_1_args},
            {"id": tool_call_2_id, "name": tool.name, "args": tool_call_2_args},
        ],
    )
    tool_message_1 = ToolMessage(
        content="Tool output 1",
        tool_call_id=tool_call_1_id,
        name=tool.name,
        artifact=LocalDataValue(value="Tool output value 1"),
    )
    tool_message_2 = ToolMessage(
        content="Tool output 2",
        tool_call_id=tool_call_2_id,
        name=tool.name,
        artifact=LocalDataValue(value="Tool output value 2"),
    )

    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=tool,
        step=Step(task="Test task", output="$output"),
    )
    base_chat_model = mock_model.to_langchain()
    result = summarizer_model.invoke({"messages": [ai_message, tool_message_1, tool_message_2]})

    assert base_chat_model.invoke.called  # type: ignore[reportFunctionMemberAccess]
    messages = base_chat_model.invoke.call_args[0][0]  # type: ignore[reportFunctionMemberAccess]
    assert messages
    assert "You are a highly skilled summarizer" in messages[0].content

    prompt_content = messages[1].content
    assert "Tool output 1" in prompt_content
    assert "Tool output 2" in prompt_content
    assert "OUTPUT_SEPARATOR" in prompt_content

    assert f"ToolCallName: {tool.name}" in prompt_content
    assert f"ToolCallArgs: {tool_call_1_args}" in prompt_content
    assert f"ToolCallArgs: {tool_call_2_args}" in prompt_content

    # Check that summaries were added to the artifact of the last message
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary == "Short summary"
    assert output_message.tool_call_id == tool_call_2_id


@pytest.mark.asyncio
async def test_summarizer_model_multiple_tool_calls_async() -> None:
    """Test the async summarizer model with multiple tool calls."""
    summary = AIMessage(content="Short summary")
    tool = AdditionTool()
    mock_model = get_mock_generative_model(response=summary)

    tool_call_1_id = "123"
    tool_call_2_id = "456"

    tool_call_1_args = {"a": 1, "b": 2}
    tool_call_2_args = {"a": 3, "b": 4}

    ai_message = AIMessage(
        content="",
        tool_calls=[
            {"id": tool_call_1_id, "name": tool.name, "args": tool_call_1_args},
            {"id": tool_call_2_id, "name": tool.name, "args": tool_call_2_args},
        ],
    )
    tool_message_1 = ToolMessage(
        content="Tool output 1",
        tool_call_id=tool_call_1_id,
        name=tool.name,
        artifact=LocalDataValue(value="Tool output value 1"),
    )
    tool_message_2 = ToolMessage(
        content="Tool output 2",
        tool_call_id=tool_call_2_id,
        name=tool.name,
        artifact=LocalDataValue(value="Tool output value 2"),
    )

    summarizer_model = StepSummarizer(
        config=get_test_config(),
        model=mock_model,
        tool=tool,
        step=Step(task="Test task", output="$output"),
    )
    base_chat_model = mock_model.to_langchain()
    result = await summarizer_model.ainvoke(
        {"messages": [ai_message, tool_message_1, tool_message_2]}
    )

    assert base_chat_model.ainvoke.called  # type: ignore[reportFunctionMemberAccess]
    messages = base_chat_model.ainvoke.call_args[0][0]  # type: ignore[reportFunctionMemberAccess]
    assert messages
    assert "You are a highly skilled summarizer" in messages[0].content

    prompt_content = messages[1].content
    assert "Tool output 1" in prompt_content
    assert "Tool output 2" in prompt_content
    assert "OUTPUT_SEPARATOR" in prompt_content

    assert f"ToolCallName: {tool.name}" in prompt_content
    assert f"ToolCallArgs: {tool_call_1_args}" in prompt_content
    assert f"ToolCallArgs: {tool_call_2_args}" in prompt_content

    # Check that summaries were added to the artifact of the last message
    output_message = result["messages"][0]
    assert isinstance(output_message, ToolMessage)
    assert output_message.artifact.summary == "Short summary"
    assert output_message.tool_call_id == tool_call_2_id
