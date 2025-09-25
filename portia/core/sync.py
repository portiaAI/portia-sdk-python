"""Synchronization utilities for executing async coroutines safely.

This module provides utilities for executing async coroutines in sync contexts,
handling event loop management and ensuring safe execution.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


class SyncAdapter:
    """Helper class for executing async coroutines safely in sync contexts.

    This class provides utilities for running async code from sync contexts,
    handling event loop management and ensuring proper cleanup.
    """

    @staticmethod
    def run(coro: Awaitable[T]) -> T:
        """Run an async coroutine and return the result.

        This method safely executes an async coroutine, handling event loop
        creation and management as needed.

        Args:
            coro: The async coroutine to execute

        Returns:
            The result of the coroutine execution

        Raises:
            RuntimeError: If there are issues with event loop management
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop is running, create a new one
            return asyncio.run(coro)
        else:
            # An event loop is already running, we can't use asyncio.run()
            # This is a common scenario in Jupyter notebooks or other async contexts
            raise RuntimeError(
                "Cannot run async coroutine: an event loop is already running. "
                "Use 'await' instead or run from a sync context."
            )

    @staticmethod
    def run_in_new_loop(coro: Awaitable[T]) -> T:
        """Run an async coroutine in a new event loop.

        This method always creates a new event loop to execute the coroutine,
        regardless of whether there's already a running loop.

        Args:
            coro: The async coroutine to execute

        Returns:
            The result of the coroutine execution
        """
        return asyncio.run(coro)

    @classmethod
    def sync_wrapper(cls, async_func: Callable[..., Awaitable[T]]) -> Callable[..., T]:
        """Create a sync wrapper for an async function.

        This decorator converts an async function into a sync function that
        can be called from sync contexts.

        Args:
            async_func: The async function to wrap

        Returns:
            A sync function that executes the async function
        """

        @functools.wraps(async_func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            coro = async_func(*args, **kwargs)
            return cls.run(coro)

        return wrapper