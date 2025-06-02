"""Tests for the estimate_tokens function."""

import pytest

from portia.token_counter import estimate_tokens


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
