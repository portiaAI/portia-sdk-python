"""Token counting utilities with fallback for offline environments."""

import concurrent.futures

import tiktoken


class TokenCounter:
    """A wrapper around tiktoken with fallback for offline environments."""

    AVERAGE_CHARS_PER_TOKEN = 5
    DEFAULT_TIMEOUT = 2.0
    _encoding = None
    _encoding_download_attempted = False

    @classmethod
    def count_tokens(cls, text: str) -> int:
        """Count tokens in text using tiktoken, with fallback to character-based estimation.

        We expect to use the fallback behaviour in offline environments where tiktoken can't
        download encodings.
        """
        if not cls._encoding_download_attempted:
            try:
                cls._encoding = cls.get_encoding_with_timeout()
            except Exception:  # noqa: BLE001
                cls._encoding = None
            cls._encoding_download_attempted = True

        if cls._encoding is not None:
            return len(cls._encoding.encode(text))
        return int(len(text) / cls.AVERAGE_CHARS_PER_TOKEN)

    @classmethod
    def get_encoding_with_timeout(cls, timeout: float | None = None) -> tiktoken.Encoding:
        """Get the encoding with a timeout for tiktoken's requests."""
        timeout = timeout or cls.DEFAULT_TIMEOUT
        executor = concurrent.futures.ThreadPoolExecutor()
        try:
            future = executor.submit(tiktoken.get_encoding, "gpt2")
            return future.result(timeout=timeout)
        finally:
            # Always shutdown without waiting to prevent hanging on timeout
            executor.shutdown(wait=False)
