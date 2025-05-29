"""Tests for the Gemini LangSmith wrapper."""

from typing import Literal
from unittest.mock import MagicMock, patch

import google.generativeai as genai
import pytest

from portia.gemini_langsmith_wrapper import (
    _process_inputs,
    _process_outputs,
    wrap_gemini,
)


@pytest.fixture
def mock_genai_model() -> genai.GenerativeModel:  # pyright: ignore[reportPrivateImportUsage]
    """Create a mock Google Generative AI model."""
    mock_model = MagicMock()
    mock_model.model_name = "gemini-1.5-pro"

    # Mock the generate_content method
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "Test response"

    mock_model.generate_content.return_value = mock_response
    return mock_model


@pytest.fixture
def mock_inputs() -> dict[str, list[dict[str, list[str]]]]:
    """Create mock inputs for testing."""
    return {
        "contents": [
            {
                "parts": [
                    "System message",
                    "User message",
                ]
            }
        ]
    }


@pytest.fixture
def mock_outputs() -> MagicMock:
    """Create mock outputs for testing."""
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "AI response"
    return mock_response


@patch("portia.gemini_langsmith_wrapper.run_helpers")
def test_wrap_gemini_successful_tracing(
    mock_run_helpers: MagicMock,
    mock_genai_model: MagicMock,
    mock_inputs: dict[str, list[dict[str, list[str]]]],
) -> None:
    """Test wrap_gemini function with successful tracing."""
    expected_result = mock_genai_model.generate_content.return_value
    mock_decorator = MagicMock()
    mock_run_helpers.traceable.return_value = mock_decorator
    mock_decorated_function = MagicMock()
    mock_decorator.return_value = mock_decorated_function
    mock_decorated_function.return_value = expected_result

    wrapped_model = wrap_gemini(mock_genai_model)
    result = wrapped_model.generate_content(mock_inputs)

    assert result == expected_result
    mock_decorator.assert_called_once()
    mock_decorated_function.assert_called_once_with(mock_inputs)

    # Verify the decorator was called with correct parameters
    mock_run_helpers.traceable.assert_called_once()
    call_kwargs = mock_run_helpers.traceable.call_args[1]
    assert call_kwargs["name"] == "GoogleGenerativeAI"
    assert call_kwargs["run_type"] == "llm"
    assert call_kwargs["process_outputs"] == _process_outputs
    assert call_kwargs["process_inputs"] == _process_inputs
    invocation_params_fn = call_kwargs["_invocation_params_fn"]
    params = invocation_params_fn({})
    assert params == {
        "ls_provider": "google_genai",
        "ls_model_name": "gemini-1.5-pro",
        "ls_model_type": "chat",
    }


@patch("portia.gemini_langsmith_wrapper.run_helpers")
def test_wrap_gemini_tracing_exception_fallback(
    mock_run_helpers: MagicMock,
    mock_genai_model: MagicMock,
    mock_inputs: dict[str, list[dict[str, list[str]]]],
) -> None:
    """Test wrap_gemini function falls back to original method when tracing fails."""
    # Store the original method before wrapping
    original_generate_content = mock_genai_model.generate_content
    mock_decorator = MagicMock()
    mock_run_helpers.traceable.return_value = mock_decorator
    mock_decorated_function = MagicMock()
    mock_decorator.return_value = mock_decorated_function
    mock_decorated_function.side_effect = Exception("Tracing error")

    wrapped_model = wrap_gemini(mock_genai_model)
    result = wrapped_model.generate_content(mock_inputs)

    # Verify the decorated function was called first and failed
    mock_decorated_function.assert_called_once_with(mock_inputs)

    # Verify the original method was called as fallback
    original_generate_content.assert_called_with(mock_inputs)

    # Verify the result is returned from the original method
    assert result == original_generate_content.return_value


def test_process_inputs(mock_inputs: dict[Literal["contents"], list[dict[str, list[str]]]]) -> None:
    """Test _process_inputs function."""
    result = _process_inputs(mock_inputs)
    assert result == {
        "messages": [
            {
                "role": "system",
                "content": "System message",
            },
            {
                "role": "user",
                "content": "User message",
            },
        ]
    }


def test_process_inputs_single_part() -> None:
    """Test _process_inputs function."""
    mock_inputs = {
        "contents": [
            {
                "parts": [
                    "User message",
                ]
            }
        ]
    }
    result = _process_inputs(mock_inputs)  # pyright: ignore[reportArgumentType]
    assert result == {
        "messages": [
            {
                "content": "User message",
            },
        ]
    }


def test_process_outputs(mock_outputs: MagicMock) -> None:
    """Test _process_outputs function."""
    result = _process_outputs(mock_outputs)
    assert result == {
        "messages": [
            {
                "role": "ai",
                "content": "AI response",
            },
        ]
    }
