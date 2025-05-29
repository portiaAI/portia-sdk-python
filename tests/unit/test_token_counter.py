"""Tests for the TokenCounter class."""

import time
import unittest.mock
from unittest.mock import MagicMock

import tiktoken

from portia.token_counter import TokenCounter


def setup_function() -> None:
    """Reset TokenCounter state before each test."""
    TokenCounter._encoding = None  # noqa: SLF001
    TokenCounter._encoding_download_attempted = False  # noqa: SLF001


def test_token_counter_happy_case() -> None:
    """Test TokenCounter when tiktoken works normally."""
    text = "Hello, world! This is a test."

    expected_count = len(tiktoken.get_encoding("gpt2").encode(text))
    actual_count = TokenCounter.count_tokens(text)

    assert actual_count == expected_count
    assert actual_count > 0
    assert TokenCounter._encoding is not None  # noqa: SLF001
    assert TokenCounter._encoding_download_attempted  # noqa: SLF001

    # Check that we don't need to get the encoding the second time
    with unittest.mock.patch("portia.token_counter.tiktoken.get_encoding") as mock_get_encoding:
        actual_count = TokenCounter.count_tokens(text)
        assert actual_count == expected_count
        mock_get_encoding.assert_not_called()
        assert TokenCounter._encoding is not None  # noqa: SLF001
        assert TokenCounter._encoding_download_attempted  # noqa: SLF001


def test_token_counter_fallback_case() -> None:
    """Test TokenCounter fallback when tiktoken fails."""
    text = "Hello, world! This is a test."

    with unittest.mock.patch("portia.token_counter.tiktoken.get_encoding") as mock_get_encoding:
        mock_get_encoding.side_effect = Exception("Tiktoken failed")

        actual_count = TokenCounter.count_tokens(text)
        expected_fallback_count = int(len(text) / TokenCounter.AVERAGE_CHARS_PER_TOKEN)

        assert actual_count == expected_fallback_count
        assert actual_count > 0


def test_token_counter_fallback_with_empty_string() -> None:
    """Test TokenCounter fallback with empty string."""
    text = ""

    with unittest.mock.patch("portia.token_counter.tiktoken.get_encoding") as mock_get_encoding:
        mock_get_encoding.side_effect = Exception("Tiktoken failed")
        assert TokenCounter.count_tokens(text) == 0


def test_token_counter_timeout_behavior() -> None:
    """Test TokenCounter timeout when tiktoken takes too long to respond."""
    text = "Hello, world! This is a test."

    def slow_get_encoding(_: str) -> tiktoken.Encoding:
        """Simulate a slow tiktoken.get_encoding call that exceeds the timeout."""
        time.sleep(1.0)  # Sleep longer than the 2-second timeout
        return MagicMock(spec=tiktoken.Encoding)

    with unittest.mock.patch(
        "portia.token_counter.tiktoken.get_encoding", side_effect=slow_get_encoding
    ):
        # This should timeout and fall back to character-based counting
        TokenCounter.DEFAULT_TIMEOUT = 0.1
        actual_count = TokenCounter.count_tokens(text)
        expected_fallback_count = int(len(text) / TokenCounter.AVERAGE_CHARS_PER_TOKEN)

        assert actual_count == expected_fallback_count
        assert actual_count > 0
        assert TokenCounter._encoding is None  # noqa: SLF001
        assert TokenCounter._encoding_download_attempted  # noqa: SLF001

    # Check that we don't bother trying to get the encoding the second time
    with unittest.mock.patch("portia.token_counter.tiktoken.get_encoding") as mock_get_encoding:
        actual_count = TokenCounter.count_tokens(text)
        expected_fallback_count = int(len(text) / TokenCounter.AVERAGE_CHARS_PER_TOKEN)

        assert actual_count == expected_fallback_count
        assert actual_count > 0
        assert TokenCounter._encoding is None  # noqa: SLF001
        assert TokenCounter._encoding_download_attempted  # noqa: SLF001
        mock_get_encoding.assert_not_called()
