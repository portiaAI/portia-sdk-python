"""tests for llm tool."""

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import HumanMessage

from portia.open_source_tools.llm_tool import LLMTool, LLMToolSchema
from portia.tool import ToolRunContext
from tests.utils import get_test_config


@pytest.fixture
def mock_execution_context() -> ToolRunContext:
    """Fixture to mock ExecutionContext."""
    return MagicMock(spec=ToolRunContext)


@pytest.fixture
def mock_llm_tool() -> LLMTool:
    """Fixture to create an instance of LLMTool."""
    return LLMTool(id="test_tool", name="Test LLM Tool")


@patch("portia.open_source_tools.llm_tool.LLMWrapper")
@patch.dict(os.environ, {"OPENAI_API_KEY": "123"})
def test_llm_tool_plan_run(
    mock_llm_wrapper: MagicMock,
    mock_execution_context: MagicMock,
    mock_llm_tool: MagicMock,
) -> None:
    """Test that LLMTool runs successfully and returns a response."""
    # Setup mock responses
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Test response content"
    mock_llm.invoke.return_value = mock_response
    mock_llm_wrapper.for_usage.return_value.to_langchain.return_value = mock_llm
    mock_execution_context.execution_context = MagicMock()
    mock_execution_context.config = get_test_config()
    mock_execution_context.plan_run_id = uuid.uuid4()
    mock_execution_context.execution_context.plan_run_context = None
    # Define task input
    task = "What is the capital of France?"

    # Run the tool
    result = mock_llm_tool.run(mock_execution_context, task)

    mock_llm.invoke.assert_called_once_with(
        [HumanMessage(content=mock_llm_tool.prompt), HumanMessage(content=task)],
    )

    # Assert the result is the expected response
    assert result == "Test response content"


def test_llm_tool_schema_valid_input() -> None:
    """Test that the LLMToolSchema correctly validates the input."""
    schema_data = {"task": "Solve a math problem"}
    schema = LLMToolSchema(**schema_data)

    assert schema.task == "Solve a math problem"


def test_llm_tool_schema_missing_task() -> None:
    """Test that LLMToolSchema raises an error if 'task' is missing."""
    with pytest.raises(ValueError):  # noqa: PT011
        LLMToolSchema()  # type: ignore  # noqa: PGH003


def test_llm_tool_initialization(mock_llm_tool: LLMTool) -> None:
    """Test that LLMTool is correctly initialized."""
    assert mock_llm_tool.id == "test_tool"
    assert mock_llm_tool.name == "Test LLM Tool"


@patch("portia.open_source_tools.llm_tool.LLMWrapper")
@patch.dict(os.environ, {"OPENAI_API_KEY": "123"})
def test_llm_tool_run_with_context(
    mock_llm_wrapper: MagicMock,
    mock_execution_context: MagicMock,
    mock_llm_tool: MagicMock,
) -> None:
    """Test that LLMTool runs successfully when a context is provided."""
    # Setup mock responses
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Test response content"
    mock_llm.invoke.return_value = mock_response
    mock_llm_wrapper.for_usage.return_value.to_langchain.return_value = mock_llm
    mock_execution_context.execution_context = MagicMock()
    mock_execution_context.config = get_test_config()
    mock_execution_context.plan_run_id = uuid.uuid4()
    # Define task and context
    mock_llm_tool.tool_context = "Context for task"
    task = "What is the capital of France?"

    # Run the tool
    result = mock_llm_tool.run(mock_execution_context, task)

    # Verify that the LLMWrapper's invoke method is called
    called_with = mock_llm.invoke.call_args_list[0].args[0]
    assert len(called_with) == 2
    assert isinstance(called_with[0], HumanMessage)
    assert isinstance(called_with[1], HumanMessage)
    assert mock_llm_tool.tool_context in called_with[1].content
    assert task in called_with[1].content
    # Assert the result is the expected response
    assert result == "Test response content"
