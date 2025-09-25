"""Unit tests for the SyncAdapter utility."""

import asyncio
import pytest
from unittest.mock import patch

from portia.core.sync import SyncAdapter


class TestSyncAdapter:
    """Test cases for SyncAdapter."""

    def test_run_simple_coroutine(self):
        """Test running a simple async coroutine."""

        async def simple_coro():
            return "test_result"

        result = SyncAdapter.run(simple_coro())
        assert result == "test_result"

    def test_run_coroutine_with_args(self):
        """Test running an async coroutine that takes arguments."""

        async def coro_with_args(value, multiplier=2):
            await asyncio.sleep(0)  # Simulate async work
            return value * multiplier

        result = SyncAdapter.run(coro_with_args(5, multiplier=3))
        assert result == 15

    def test_run_raises_runtime_error_with_running_loop(self):
        """Test that run raises RuntimeError when there's already a running loop."""

        async def test_coro():
            return "test"

        # Mock get_running_loop to simulate an already running loop
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_get_loop.return_value = "mock_loop"

            with pytest.raises(RuntimeError) as exc_info:
                SyncAdapter.run(test_coro())

            assert "Cannot run async coroutine" in str(exc_info.value)
            assert "event loop is already running" in str(exc_info.value)

    def test_run_in_new_loop(self):
        """Test running a coroutine in a new event loop."""

        async def simple_coro():
            return "new_loop_result"

        result = SyncAdapter.run_in_new_loop(simple_coro())
        assert result == "new_loop_result"

    def test_run_in_new_loop_with_complex_async_operations(self):
        """Test running complex async operations in new loop."""

        async def complex_coro():
            await asyncio.sleep(0.01)  # Small sleep to test async behavior
            results = []
            for i in range(3):
                await asyncio.sleep(0.001)
                results.append(i * 2)
            return sum(results)

        result = SyncAdapter.run_in_new_loop(complex_coro())
        assert result == 6  # 0 + 2 + 4 = 6

    def test_sync_wrapper_decorator(self):
        """Test the sync_wrapper decorator functionality."""

        async def async_function(x, y=1):
            await asyncio.sleep(0)
            return x + y

        # Wrap the async function
        sync_function = SyncAdapter.sync_wrapper(async_function)

        # Test that the wrapper returns the correct result
        result = sync_function(5, y=3)
        assert result == 8

        # Test with positional arguments
        result = sync_function(10, 20)
        assert result == 30

    def test_sync_wrapper_preserves_function_metadata(self):
        """Test that sync_wrapper preserves the original function's metadata."""

        async def documented_function(arg1, arg2):
            """This is a test function with documentation."""
            return arg1 + arg2

        sync_function = SyncAdapter.sync_wrapper(documented_function)

        # Check that function name and docstring are preserved
        assert sync_function.__name__ == "documented_function"
        assert sync_function.__doc__ == "This is a test function with documentation."

    def test_sync_wrapper_handles_exceptions(self):
        """Test that sync_wrapper properly propagates exceptions."""

        async def failing_function():
            raise ValueError("Test exception")

        sync_function = SyncAdapter.sync_wrapper(failing_function)

        with pytest.raises(ValueError) as exc_info:
            sync_function()

        assert str(exc_info.value) == "Test exception"

    def test_sync_wrapper_with_no_event_loop_running(self):
        """Test sync_wrapper when no event loop is running."""

        async def async_add(a, b):
            return a + b

        sync_add = SyncAdapter.sync_wrapper(async_add)
        result = sync_add(3, 4)
        assert result == 7

    @patch("asyncio.get_running_loop")
    def test_sync_wrapper_with_running_loop_raises_error(self, mock_get_loop):
        """Test sync_wrapper raises error when event loop is running."""
        mock_get_loop.return_value = "mock_loop"

        async def async_function():
            return "test"

        sync_function = SyncAdapter.sync_wrapper(async_function)

        with pytest.raises(RuntimeError) as exc_info:
            sync_function()

        assert "Cannot run async coroutine" in str(exc_info.value)