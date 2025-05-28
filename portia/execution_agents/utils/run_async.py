"""Helper function to smooth over running async functions in a sync context."""

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a co-routine sync."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop is running: safe to use asyncio.run
        return asyncio.run(coro)

    if loop.is_running():
        # Loop is already running — run coro in a new loop in a thread
        result = []
        exception = []

        def thread_worker() -> None:
            """Run a coroutine in a thread."""
            new_loop = None
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result.append(new_loop.run_until_complete(coro))
            except Exception as e:  # noqa: BLE001
                exception.append(e)
            finally:
                if new_loop is not None:
                    new_loop.close()

        thread = threading.Thread(target=thread_worker)
        thread.start()
        thread.join()

        if exception:
            raise exception[0]
        return result[0]

    # Loop exists but not running (should never happen but for completeness)
    return loop.run_until_complete(coro)
