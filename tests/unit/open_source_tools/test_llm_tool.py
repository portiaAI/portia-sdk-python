"""tests for llm tool."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from portia.model import Message
from portia.open_source_tools.llm_tool import LLMTool, LLMToolSchema
from portia.tool import ToolRunContext


@pytest.fixture
def mock_llm_tool() -> LLMTool:
    """Fixture to create an instance of LLMTool."""
    return LLMTool(id="test_tool", name="Test LLM Tool")


def test_llm_tool_plan_run(
    mock_llm_tool: LLMTool,
    mock_tool_run_context: ToolRunContext,
    mock_model: MagicMock,
) -> None:
    """Test that LLMTool runs successfully and returns a response."""
    # Setup mock responses
    mock_model.get_response.return_value = Message(role="user", content="Test response content")
    # Define task input
    task = "What is the capital of France?"

    # Run the tool
    result = mock_llm_tool.run(mock_tool_run_context, task)

    mock_model.get_response.assert_called_once_with(
        [Message(role="user", content=mock_llm_tool.prompt), Message(role="user", content=task)],
    )

    # Assert the result is the expected response
    assert result == "Test response content"


def test_llm_tool_structured_output_run(
    mock_llm_tool: LLMTool,
    mock_tool_run_context: ToolRunContext,
    mock_model: MagicMock,
) -> None:
    """Test that LLMTool runs successfully and returns a response."""

    class TestStructuredOutput(BaseModel):
        capital: str

    # Setup mock responses
    mock_model.get_structured_response.return_value = TestStructuredOutput(capital="Paris")
    # Define task input
    task = "What is the capital of France?"

    mock_llm_tool.structured_output_schema = TestStructuredOutput

    # Run the tool
    mock_llm_tool.structured_output_schema = TestStructuredOutput
    result = mock_llm_tool.run(mock_tool_run_context, task)

    mock_model.get_structured_response.assert_called_once_with(
        [Message(role="user", content=mock_llm_tool.prompt), Message(role="user", content=task)],
        TestStructuredOutput,
    )

    # Assert the result is the expected response
    assert result == TestStructuredOutput(capital="Paris")


def test_llm_tool_schema_valid_input() -> None:
    """Test that the LLMToolSchema correctly validates the input."""
    schema_data = {"task": "Solve a math problem", "task_data": ["1 + 1 = 2"]}
    schema = LLMToolSchema(**schema_data)

    assert schema.task == "Solve a math problem"
    assert schema.task_data == ["1 + 1 = 2"]


def test_llm_tool_schema_missing_task() -> None:
    """Test that LLMToolSchema raises an error if 'task' is missing."""
    with pytest.raises(ValueError):  # noqa: PT011
        LLMToolSchema()  # type: ignore  # noqa: PGH003


def test_llm_tool_initialization(mock_llm_tool: LLMTool) -> None:
    """Test that LLMTool is correctly initialized."""
    assert mock_llm_tool.id == "test_tool"
    assert mock_llm_tool.name == "Test LLM Tool"


def test_llm_tool_run_with_context(
    mock_llm_tool: LLMTool,
    mock_tool_run_context: ToolRunContext,
    mock_model: MagicMock,
) -> None:
    """Test that LLMTool runs successfully when a context is provided."""
    # Setup mock responses
    mock_model.get_response.return_value = Message(role="user", content="Test response content")

    # Define task and context
    mock_llm_tool.tool_context = "Context for task"
    task = "What is the capital of France?"

    # Run the tool
    result = mock_llm_tool.run(mock_tool_run_context, task)

    # Verify that the Model's get_response method is called
    called_with = mock_model.get_response.call_args_list[0].args[0]
    assert len(called_with) == 2
    assert isinstance(called_with[0], Message)
    assert isinstance(called_with[1], Message)
    assert mock_llm_tool.tool_context in called_with[1].content
    assert task in called_with[1].content
    # Assert the result is the expected response
    assert result == "Test response content"


def test_process_task_data_with_string() -> None:
    """Test that process_task_data correctly handles string input."""
    result = LLMTool.process_task_data("String data")
    assert result == "String data"


def test_process_task_data_with_list() -> None:
    """Test that process_task_data correctly handles list input."""
    result = LLMTool.process_task_data(["Item 1", "Item 2"])
    assert result == "Item 1\nItem 2"


def test_process_task_data_with_none() -> None:
    """Test that process_task_data correctly handles None input."""
    result = LLMTool.process_task_data(None)
    assert result == ""


def test_process_task_data_with_complex_objects() -> None:
    """Test that process_task_data correctly handles complex objects."""

    class TestObject:
        def __str__(self) -> str:
            return "TestObject"

    result = LLMTool.process_task_data([TestObject(), {"nested": "value"}])
    assert result == "TestObject\n{'nested': 'value'}"


def test_process_task_data_with_dict() -> None:
    """Test that process_task_data correctly handles dictionary input."""
    result = LLMTool.process_task_data({"key": "value", "key2": "value2"})
    assert result == "key: value\nkey2: value2"


# Async tests for LLMTool.arun function
@pytest.mark.asyncio
async def test_llm_tool_async_plan_run(
    mock_llm_tool: LLMTool,
    mock_tool_run_context: ToolRunContext,
    mock_model: MagicMock,
) -> None:
    """Test that LLMTool runs successfully asynchronously and returns a response."""
    # Setup mock responses
    mock_model.aget_response.return_value = Message(
        role="user", content="Test async response content"
    )
    # Define task input
    task = "What is the capital of France?"

    # Run the tool asynchronously
    result = await mock_llm_tool.arun(mock_tool_run_context, task)

    mock_model.aget_response.assert_called_once_with(
        [Message(role="user", content=mock_llm_tool.prompt), Message(role="user", content=task)],
    )

    # Assert the result is the expected response
    assert result == "Test async response content"


@pytest.mark.asyncio
async def test_llm_tool_async_structured_output_run(
    mock_llm_tool: LLMTool,
    mock_tool_run_context: ToolRunContext,
    mock_model: MagicMock,
) -> None:
    """Test that LLMTool runs successfully asynchronously and returns a structured response."""

    class TestStructuredOutput(BaseModel):
        capital: str

    # Setup mock responses
    mock_model.aget_structured_response.return_value = TestStructuredOutput(capital="Paris")
    # Define task input
    task = "What is the capital of France?"

    mock_llm_tool.structured_output_schema = TestStructuredOutput

    # Run the tool asynchronously
    result = await mock_llm_tool.arun(mock_tool_run_context, task)

    mock_model.aget_structured_response.assert_called_once_with(
        [Message(role="user", content=mock_llm_tool.prompt), Message(role="user", content=task)],
        TestStructuredOutput,
    )

    # Assert the result is the expected response
    assert result == TestStructuredOutput(capital="Paris")


@pytest.mark.asyncio
async def test_llm_tool_async_run_with_context(
    mock_llm_tool: LLMTool,
    mock_tool_run_context: ToolRunContext,
    mock_model: MagicMock,
) -> None:
    """Test that LLMTool runs successfully asynchronously when a context is provided."""
    # Setup mock responses
    mock_model.aget_response.return_value = Message(
        role="user", content="Test async response content"
    )

    # Define task and context
    mock_llm_tool.tool_context = "Context for task"
    task = "What is the capital of France?"

    # Run the tool asynchronously
    result = await mock_llm_tool.arun(mock_tool_run_context, task)

    # Verify that the Model's aget_response method is called
    called_with = mock_model.aget_response.call_args_list[0].args[0]
    assert len(called_with) == 2
    assert isinstance(called_with[0], Message)
    assert isinstance(called_with[1], Message)
    assert mock_llm_tool.tool_context in called_with[1].content
    assert task in called_with[1].content
    # Assert the result is the expected response
    assert result == "Test async response content"


@pytest.mark.asyncio
async def test_llm_tool_async_run_with_task_data(
    mock_llm_tool: LLMTool,
    mock_tool_run_context: ToolRunContext,
    mock_model: MagicMock,
) -> None:
    """Test that LLMTool runs successfully asynchronously with task data."""
    # Setup mock responses
    mock_model.aget_response.return_value = Message(
        role="user", content="Test async response with data"
    )
    # Define task input with task data
    task = "Analyze this data"
    task_data = ["Data point 1", "Data point 2", {"key": "value"}]

    # Run the tool asynchronously
    result = await mock_llm_tool.arun(mock_tool_run_context, task, task_data)

    # Verify that the Model's aget_response method is called with the correct messages
    called_with = mock_model.aget_response.call_args_list[0].args[0]
    assert len(called_with) == 2
    assert isinstance(called_with[0], Message)
    assert isinstance(called_with[1], Message)
    assert "Task data: Data point 1\nData point 2\n{'key': 'value'}" in called_with[1].content
    assert task in called_with[1].content
    # Assert the result is the expected response
    assert result == "Test async response with data"


@pytest.mark.asyncio
async def test_llm_tool_async_run_with_string_task_data(
    mock_llm_tool: LLMTool,
    mock_tool_run_context: ToolRunContext,
    mock_model: MagicMock,
) -> None:
    """Test that LLMTool runs successfully asynchronously with string task data."""
    # Setup mock responses
    mock_model.aget_response.return_value = Message(
        role="user", content="Test async response with string data"
    )
    # Define task input with string task data
    task = "Analyze this text"
    task_data = "This is a string of text data"

    # Run the tool asynchronously
    result = await mock_llm_tool.arun(mock_tool_run_context, task, task_data)

    # Verify that the Model's aget_response method is called with the correct messages
    called_with = mock_model.aget_response.call_args_list[0].args[0]
    assert len(called_with) == 2
    assert isinstance(called_with[0], Message)
    assert isinstance(called_with[1], Message)
    assert "Task data: This is a string of text data" in called_with[1].content
    assert task in called_with[1].content
    # Assert the result is the expected response
    assert result == "Test async response with string data"


@pytest.mark.asyncio
async def test_llm_tool_async_run_with_none_task_data(
    mock_llm_tool: LLMTool,
    mock_tool_run_context: ToolRunContext,
    mock_model: MagicMock,
) -> None:
    """Test that LLMTool runs successfully asynchronously with None task data."""
    # Setup mock responses
    mock_model.aget_response.return_value = Message(
        role="user", content="Test async response with no data"
    )
    # Define task input with None task data
    task = "Answer this question"

    # Run the tool asynchronously
    result = await mock_llm_tool.arun(mock_tool_run_context, task, None)

    # Verify that the Model's aget_response method is called with the correct messages
    called_with = mock_model.aget_response.call_args_list[0].args[0]
    assert len(called_with) == 2
    assert isinstance(called_with[0], Message)
    assert isinstance(called_with[1], Message)
    assert "Task data:" not in called_with[1].content  # Should not include task data section
    assert task in called_with[1].content
    # Assert the result is the expected response
    assert result == "Test async response with no data"
