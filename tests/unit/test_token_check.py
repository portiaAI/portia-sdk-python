"""Tests for the estimate_tokens function."""

from unittest.mock import MagicMock

import pytest

from portia.token_check import estimate_tokens, exceeds_context_threshold


@pytest.mark.parametrize(
    ("text", "expected_tokens"),
    [
        ("", 0),
        ("Hello, world! This is a test.", 5),
        ('{"name": "John", "age": 30, "city": "New York"}', 9),
    ],
)
def test_estimate_tokens(text: str, expected_tokens: int) -> None:
    """Test estimate_tokens function with various input cases."""
    actual_tokens = estimate_tokens(text)
    assert actual_tokens == expected_tokens


@pytest.mark.parametrize(
    ("value", "threshold_percentage", "expected_result"),
    [
        ("short value", None, False),
        ("a very long value" * 1000, None, True),
        ("token" * 700, 0.5, True),
        ("token" * 300, 0.5, False),
        (None, 1000, False),
    ],
)
def test_exceeds_context_threshold(
    value: str, threshold_percentage: int | None, expected_result: bool
) -> None:
    """Test exceeds_context_threshold function with various input cases."""
    model = MagicMock()
    model.get_context_window_size.return_value = 1000
    if threshold_percentage:
        assert exceeds_context_threshold(value, model, threshold_percentage) == expected_result
    else:
        assert exceeds_context_threshold(value, model) == expected_result
