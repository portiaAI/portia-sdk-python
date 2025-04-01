"""Tests for image understanding tool."""

from unittest.mock import MagicMock

import pytest
from langchain.schema import HumanMessage
from pydantic import ValidationError

from portia.open_source_tools.image_understanding_tool import (
    ImageUnderstandingTool,
    ImageUnderstandingToolSchema,
)
from portia.tool import ToolRunContext


@pytest.fixture
def mock_image_understanding_tool() -> ImageUnderstandingTool:
    """Fixture to create an instance of ImageUnderstandingTool."""
    return ImageUnderstandingTool(id="test_tool", name="Test Image Understanding Tool")


def test_image_understanding_tool_run_url(
    mock_tool_run_context: ToolRunContext,
    mock_image_understanding_tool: ImageUnderstandingTool,
    mock_model: MagicMock,
) -> None:
    """Test that ImageUnderstandingTool runs successfully and returns a response."""
    # Setup mock responses
    mock_response = MagicMock()
    mock_response.content = "Test response content"
    mock_model.to_langchain.return_value.invoke.return_value = mock_response

    # Define task input
    schema_data = {
        "task": "What is the capital of France?",
        "image_url": "https://example.com/image.png",
    }

    # Run the tool
    result = mock_image_understanding_tool.run(mock_tool_run_context, **schema_data)

    mock_model.to_langchain.return_value.invoke.assert_called_once_with(
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


def test_image_understanding_tool_schema_valid_input() -> None:
    """Test that the LLMToolSchema correctly validates the input."""
    schema_data = {
        "task": "Solve a math problem in this image",
        "image_url": "https://example.com/image.png",
    }
    schema = ImageUnderstandingToolSchema(**schema_data)

    assert schema.task == "Solve a math problem in this image"
    assert schema.image_url == "https://example.com/image.png"


def test_image_understanding_tool_schema_missing_task() -> None:
    """Test that LLMToolSchema raises an error if 'task' is missing."""
    with pytest.raises(ValidationError):
        ImageUnderstandingToolSchema(image_url="https://example.com/image.png")  # type: ignore  # noqa: PGH003


def test_image_understanding_tool_schema_missing_image_url_and_file() -> None:
    """Test that LLMToolSchema raises an error if 'image_url' and 'image_file' are missing."""
    with pytest.raises(ValidationError):
        ImageUnderstandingToolSchema(task="Solve a math problem in this image")  # type: ignore  # noqa: PGH003


def test_image_understanding_tool_schema_both_image_url_and_file() -> None:
    """Test that LLMToolSchema raises an error if 'image_url' and 'image_file' are provided."""
    with pytest.raises(ValidationError):
        ImageUnderstandingToolSchema(
            task="Solve a math problem in this image",
            image_url="https://example.com/image.png",
            image_file="image.png",
        )  # type: ignore  # noqa: PGH003


def test_image_understanding_tool_initialization(
    mock_image_understanding_tool: ImageUnderstandingTool,
) -> None:
    """Test that LLMTool is correctly initialized."""
    assert mock_image_understanding_tool.id == "test_tool"
    assert mock_image_understanding_tool.name == "Test Image Understanding Tool"


def test_image_understanding_tool_run_with_context(
    mock_model: MagicMock,
    mock_tool_run_context: ToolRunContext,
    mock_image_understanding_tool: ImageUnderstandingTool,
) -> None:
    """Test that ImageUnderstandingTool runs successfully when a context is provided."""
    # Setup mock responses
    mock_response = MagicMock()
    mock_response.content = "Test response content"
    mock_model.to_langchain.return_value.invoke.return_value = mock_response
    # Define task and context
    mock_image_understanding_tool.tool_context = "Context for task"
    exec_context = mock_tool_run_context.execution_context
    exec_context.plan_run_context = "Context from plan run context"
    schema_data = {
        "task": "What is the capital of France?",
        "image_url": "https://example.com/map.png",
    }

    # Run the tool
    result = mock_image_understanding_tool.run(mock_tool_run_context, **schema_data)

    # Verify that the Models's to_langchain().invoke method is called
    called_with = mock_model.to_langchain.return_value.invoke.call_args_list[0].args[0]
    assert len(called_with) == 2
    assert isinstance(called_with[0], HumanMessage)
    assert isinstance(called_with[1], HumanMessage)
    assert mock_image_understanding_tool.tool_context in called_with[1].content[0]["text"]
    assert exec_context.plan_run_context in called_with[1].content[0]["text"]
    # Assert the result is the expected response
    assert result == "Test response content"
