"""Test genai wrapper."""

from collections.abc import Callable
from unittest.mock import Mock, patch

import pytest
try:
    from google.genai import types  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    types = None  # type: ignore[assignment]

from portia.gemini_langsmith_wrapper import (
    _extract_parts,
    _get_ls_params,
    _process_inputs,
    _process_outputs,
    wrap_gemini,
)

# ------------------------
# Tests for _get_ls_params
# ------------------------


def test_get_ls_params() -> None:
    """Check get params."""
    params = _get_ls_params("gemini-pro", {})
    assert params == {
        "ls_provider": "google_genai",
        "ls_model_name": "gemini-pro",
        "ls_model_type": "chat",
    }


# ------------------------
# Tests for _process_outputs
# ------------------------


def test_process_outputs_valid() -> None:
    """Check valid output."""
    candidate = types.Candidate(content=types.Content(parts=[types.Part(text="Hello world")]))
    outputs = types.GenerateContentResponse(candidates=[candidate])
    result = _process_outputs(outputs)
    assert result == {"messages": [{"role": "ai", "content": "Hello world"}]}


def test_process_outputs_empty() -> None:
    """Check empty output."""
    outputs = types.GenerateContentResponse(candidates=[])
    result = _process_outputs(outputs)
    assert result == {"messages": []}


# ------------------------
# Tests for _extract_parts
# ------------------------


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        (["hi", {"text": "there"}, types.Part(text="!")], ["hi", "there", "!"]),
        (
            types.Content(parts=[types.Part(text="a"), types.Part(text="b"), types.Part(text="c")]),
            ["a", "b", "c"],
        ),
        (types.Part(text="foo"), ["foo"]),
        ({"parts": "bar"}, ["bar"]),
        ("baz", ["baz"]),
        (None, []),
        (types.Content(parts=None), []),
    ],
)
def test_extract_parts(
    input_value: types.ContentUnion | types.ContentUnionDict,
    expected: list[str],
) -> None:
    """Check extract parts."""
    assert _extract_parts(input_value) == expected


# ------------------------
# Tests for _process_inputs
# ------------------------


def test_process_inputs_with_two_parts() -> None:
    """Check with two parts."""
    inputs = {
        "contents": [
            types.Content(parts=[types.Part(text="system msg"), types.Part(text="user msg")]),
        ]
    }
    result = _process_inputs(inputs)  # type: ignore  # noqa: PGH003
    assert result == {
        "messages": [
            {"role": "system", "content": "system msg"},
            {"role": "user", "content": "user msg"},
        ]
    }


def test_process_inputs_single_part() -> None:
    """Check with single input."""
    inputs = {
        "contents": [
            types.Content(parts=[types.Part(text="hello msg")]),
        ]
    }
    result = _process_inputs(inputs)  # type: ignore  # noqa: PGH003
    assert result == {"messages": [{"content": "hello msg"}]}


def test_process_inputs_no_list() -> None:
    """Check with single input."""
    inputs = {
        "contents": types.Content(parts=[types.Part(text="hello msg")]),
    }
    result = _process_inputs(inputs)  # type: ignore  # noqa: PGH003
    assert result == {"messages": [{"content": "hello msg"}]}


def test_process_inputs_invalid() -> None:
    """Check no error on invalid."""
    result = _process_inputs({})
    assert result == {"messages": []}


# ------------------------
# Tests for wrap_gemini
# ------------------------


@pytest.fixture
def fake_client() -> tuple[Mock, Mock]:
    """Mock Client."""
    mock_client = Mock()
    mock_model_interface = Mock()
    mock_generate_content = Mock(return_value="original result")
    mock_model_interface.generate_content = mock_generate_content
    mock_client.models = mock_model_interface
    return mock_client, mock_generate_content


@patch("portia.gemini_langsmith_wrapper.run_helpers.traceable")
def test_wrap_gemini_traces_and_calls_original(traceable_mock: Mock, fake_client: Mock) -> None:
    """Check success."""
    client, original_generate_content = fake_client

    traced_func = Mock(return_value="traced result")
    traceable_mock.return_value = lambda _: traced_func

    wrapped_client = wrap_gemini(client)

    result = wrapped_client.models.generate_content(
        model="gemini-pro", contents=["Say hello!"], config={"temperature": 0.5}
    )

    assert result == "traced result"
    traced_func.assert_called_once_with("gemini-pro", ["Say hello!"], {"temperature": 0.5})
    original_generate_content.assert_not_called()


@patch("portia.gemini_langsmith_wrapper.run_helpers.traceable")
@patch("portia.gemini_langsmith_wrapper.logger")
def test_wrap_gemini_falls_back_on_trace_error(
    trace_logger: Mock,
    traceable_mock: Mock,
    fake_client: Mock,
) -> None:
    """Check error."""
    client, original_generate_content = fake_client

    def raise_in_tracing(
        run_type: str,  # noqa: ARG001
        name: str,  # noqa: ARG001
        process_inputs: Callable[[dict], dict],  # noqa: ARG001
        process_outputs: Callable[..., dict],  # noqa: ARG001
        _invocation_params_fn: Callable[[dict], dict],
    ) -> Callable:
        """Fake an error."""

        def inner(*args, **kwargs) -> None:  # noqa: ANN002, ANN003, ARG001
            """Raise error."""
            raise ValueError("tracing broke")

        return inner

    traceable_mock.side_effect = raise_in_tracing
    original_generate_content.return_value = "original fallback result"

    wrapped_client = wrap_gemini(client)

    result = wrapped_client.models.generate_content(
        model="gemini-pro",
        contents=["Fail gracefully?"],
        config=None,
    )

    assert result == "original fallback result"
    original_generate_content.assert_called_once_with("gemini-pro", ["Fail gracefully?"], None)
    trace_logger().error.assert_called_once()
