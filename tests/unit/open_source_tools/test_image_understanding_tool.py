"""Tests for image understanding tool."""

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import HumanMessage

from portia.open_source_tools.image_understanding_tool import (
    ImageUnderstandingTool,
    ImageUnderstandingToolSchema,
)
from portia.tool import ToolRunContext
from tests.utils import get_test_config


@pytest.fixture
def mock_execution_context() -> ToolRunContext:
    """Fixture to mock ExecutionContext."""
    return MagicMock(spec=ToolRunContext)


@pytest.fixture
def mock_image_understanding_tool() -> ImageUnderstandingTool:
    """Fixture to create an instance of ImageUnderstandingTool."""
    return ImageUnderstandingTool(id="test_tool", name="Test Image Understanding Tool")


@patch("portia.open_source_tools.image_understanding_tool.LLMWrapper")
@patch.dict(os.environ, {"OPENAI_API_KEY": "123"})
def test_image_understanding_tool_run(
    mock_llm_wrapper: MagicMock,
    mock_execution_context: MagicMock,
    mock_image_understanding_tool: MagicMock,
) -> None:
    """Test that ImageUnderstandingTool runs successfully and returns a response."""
    # Setup mock responses
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Test response content"
    mock_llm.invoke.return_value = mock_response
    mock_llm_wrapper.return_value.to_langchain.return_value = mock_llm
    mock_execution_context.execution_context = MagicMock()
    mock_execution_context.config = get_test_config()
    mock_execution_context.plan_run_id = uuid.uuid4()
    mock_execution_context.execution_context.plan_run_context = None
    # Define task input
    schema_data = {
        "task": "What is the capital of France?",
        "image_url": "https://example.com/image.png",
    }

    # Run the tool
    result = mock_image_understanding_tool.run(mock_execution_context, **schema_data)

    mock_llm.invoke.assert_called_once_with(
        [
            HumanMessage(content=mock_image_understanding_tool.prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": schema_data["task"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": schema_data["image_url"]},
                    },
                ],
            ),
        ],
    )

    # Assert the result is the expected response
    assert result == "Test response content"


def test_llm_tool_schema_valid_input() -> None:
    """Test that the LLMToolSchema correctly validates the input."""
    schema_data = {
        "task": "Solve a math problem in this image",
        "image_url": "https://example.com/image.png",
    }
    schema = ImageUnderstandingToolSchema(**schema_data)

    assert schema.task == "Solve a math problem in this image"
    assert schema.image_url == "https://example.com/image.png"


def test_llm_tool_schema_missing_task() -> None:
    """Test that LLMToolSchema raises an error if 'task' is missing."""
    with pytest.raises(ValueError):  # noqa: PT011
        ImageUnderstandingToolSchema()  # type: ignore  # noqa: PGH003


def test_llm_tool_initialization(mock_image_understanding_tool: ImageUnderstandingTool) -> None:
    """Test that LLMTool is correctly initialized."""
    assert mock_image_understanding_tool.id == "test_tool"
    assert mock_image_understanding_tool.name == "Test Image Understanding Tool"


@patch("portia.open_source_tools.image_understanding_tool.LLMWrapper")
@patch.dict(os.environ, {"OPENAI_API_KEY": "123"})
def test_llm_tool_run_with_context(
    mock_llm_wrapper: MagicMock,
    mock_execution_context: MagicMock,
    mock_image_understanding_tool: MagicMock,
) -> None:
    """Test that ImageUnderstandingTool runs successfully when a context is provided."""
    # Setup mock responses
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Test response content"
    mock_llm.invoke.return_value = mock_response
    mock_llm_wrapper.return_value.to_langchain.return_value = mock_llm
    mock_execution_context.execution_context = MagicMock()
    mock_execution_context.config = get_test_config()
    mock_execution_context.plan_run_id = uuid.uuid4()
    # Define task and context
    mock_image_understanding_tool.tool_context = "Context for task"
    schema_data = {
        "task": "What is the capital of France?",
        "image_url": "https://example.com/map.png",
    }

    # Run the tool
    result = mock_image_understanding_tool.run(mock_execution_context, **schema_data)

    # Verify that the LLMWrapper's invoke method is called
    called_with = mock_llm.invoke.call_args_list[0].args[0]
    assert len(called_with) == 2
    assert isinstance(called_with[0], HumanMessage)
    assert isinstance(called_with[1], HumanMessage)
    assert mock_image_understanding_tool.tool_context in called_with[1].content[0]["text"]
    # Assert the result is the expected response
    assert result == "Test response content"
