"""Tests for run_async helper."""

import asyncio
from pdb import run
from unittest import mock

import pytest

from portia.execution_agents.utils.run_async import run_async


async def sample_coro(x: int) -> int:
    """Test coro."""
    await asyncio.sleep(0.01)
    return x + 1


async def failing_coro() -> str:
    """Test error coro."""
    await asyncio.sleep(0.01)
    raise ValueError("Boom")


def test_run_async_no_loop() -> None:
    """Test no loop."""
    result = run_async(sample_coro(1))
    assert result == 2


@pytest.mark.asyncio
async def test_run_async_with_loop() -> None:
    """Test with existing loop."""

    # Run in a thread so we can test run_async from within an already-running loop
    loop_result = run_async(sample_coro(2))
    assert loop_result == 3


@pytest.mark.asyncio
async def test_run_async_exception_propagation() -> None:
    """Test with existing loop."""
    with pytest.raises(ValueError, match="Boom"):
        run_async(failing_coro())


def test_run_async_with_existing_non_running_loop() -> None:
    """Test when loop isn't running."""
    mock_loop = mock.Mock()
    mock_loop.is_running.return_value = False
    mock_loop.run_until_complete.return_value = 11

    with mock.patch("asyncio.get_running_loop", return_value=mock_loop):
        result = run_async(sample_coro(10))
        assert result == 11
