"""Test genai wrapper."""

import pytest
from google.genai import types

from portia.gemini_langsmith_wrapper import (
    _extract_parts,
    _get_ls_params,
    _process_inputs,
    _process_outputs,
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
        ({"text": "bar"}, ["bar"]),
        ("baz", ["baz"]),
        (None, []),
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


def test_process_inputs_invalid() -> None:
    """Check no error on invalid."""
    result = _process_inputs({})
    assert result == {"messages": []}
