"""Token counting utilities with fallback for offline environments."""

AVERAGE_CHARS_PER_TOKEN = 5


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a string using character-based estimation.

    We used to do a proper count using tiktoken, but that loads encodings from the internet at
    runtime, which doens't work in environments where we don't have internet access / where network
    access is locked down. As our current usages only require an estimate, this suffices for now.
    """
    return int(len(text) / AVERAGE_CHARS_PER_TOKEN)
