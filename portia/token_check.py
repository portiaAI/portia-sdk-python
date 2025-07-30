"""Token counting utilities with fallback for offline environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from portia.model import GenerativeModel

AVERAGE_CHARS_PER_TOKEN = 5


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a string using character-based estimation.

    We used to do a proper count using tiktoken, but that loads encodings from the internet at
    runtime, which doens't work in environments where we don't have internet access / where network
    access is locked down. As our current usages only require an estimate, this suffices for now.
    """
    return int(len(text) / AVERAGE_CHARS_PER_TOKEN)


def exceeds_context_threshold(
    value: Any,  # noqa: ANN401
    model: GenerativeModel,
    threshold_percentage: float = 1,
) -> bool:
    """Check if a value is under a given threshold percentage of a model's context window size.

    Args:
        value: The value to check (will be converted to string for token estimation)
        model: The generative model to get context window size from
        threshold_percentage: A percentage threshold to apply. For example, 0.9 means that this will
          return True if the value exceeds 90% of the context window size.

    Returns:
        bool: True if the estimated tokens are less than the threshold, False otherwise

    """
    value_str = str(value) if value is not None else ""
    estimated_tokens = estimate_tokens(value_str)
    context_window_size = model.get_context_window_size()
    threshold_tokens = int(context_window_size * threshold_percentage)
    return estimated_tokens > threshold_tokens
