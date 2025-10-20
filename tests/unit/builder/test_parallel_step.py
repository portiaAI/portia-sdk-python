"""Test the parallel step."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from portia.builder.conditionals import ConditionalBlock
from portia.builder.invoke_tool_step import InvokeToolStep
from portia.builder.llm_step import LLMStep
from portia.builder.loops import LoopBlock
from portia.builder.parallel_step import ParallelStep
from portia.execution_agents.output import LocalDataValue
from portia.plan import Step as PlanStep


def test_parallel_step_initialization() -> None:
    """Test ParallelStep initialization with child steps."""
    step1 = InvokeToolStep(tool="tool1", step_name="step1")
    step2 = InvokeToolStep(tool="tool2", step_name="step2")

    parallel_step = ParallelStep(steps=[step1, step2], step_name="parallel_execution")

    assert parallel_step.step_name == "parallel_execution"
    assert len(parallel_step.steps) == 2
    assert parallel_step.steps[0] is step1
    assert parallel_step.steps[1] is step2


def test_parallel_step_str() -> None:
    """Test ParallelStep str method."""
    step1 = InvokeToolStep(tool="tool1", step_name="step1")
    step2 = LLMStep(task="test task", step_name="step2")

    parallel_step = ParallelStep(steps=[step1, step2], step_name="parallel_execution")

    result = str(parallel_step)
    assert "ParallelStep" in result
    assert "InvokeToolStep" in result
    assert "LLMStep" in result


@pytest.mark.asyncio
async def test_parallel_step_run_success() -> None:
    """Test ParallelStep run with successful execution of all child steps."""
    # Create mock steps
    step1 = Mock()
    step1.run = AsyncMock(return_value="result1")

    step2 = Mock()
    step2.run = AsyncMock(return_value="result2")

    step3 = Mock()
    step3.run = AsyncMock(return_value="result3")

    parallel_step = ParallelStep(steps=[step1, step2, step3], step_name="parallel_execution")

    mock_run_data = Mock()

    result = await parallel_step.run(mock_run_data)

    assert result == ["result1", "result2", "result3"]
    step1.run.assert_called_once_with(mock_run_data)
    step2.run.assert_called_once_with(mock_run_data)
    step3.run.assert_called_once_with(mock_run_data)


@pytest.mark.asyncio
async def test_parallel_step_run_with_real_steps() -> None:
    """Test ParallelStep run with real InvokeToolStep instances."""
    # Create real steps but with mocked execution
    step1 = InvokeToolStep(tool="tool1", step_name="step1", args={"arg1": "value1"})
    step2 = InvokeToolStep(tool="tool2", step_name="step2", args={"arg2": "value2"})

    parallel_step = ParallelStep(steps=[step1, step2], step_name="parallel_execution")

    mock_run_data = Mock()
    mock_tool1 = Mock()
    mock_tool1.structured_output_schema = None
    mock_output1 = Mock()
    mock_output1.get_value.return_value = "tool1_result"
    mock_tool1._arun = AsyncMock(return_value=mock_output1)

    mock_tool2 = Mock()
    mock_tool2.structured_output_schema = None
    mock_output2 = Mock()
    mock_output2.get_value.return_value = "tool2_result"
    mock_tool2._arun = AsyncMock(return_value=mock_output2)

    with (
        patch("portia.builder.invoke_tool_step.ToolCallWrapper.from_tool_id") as mock_get_tool,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
    ):

        def get_tool_side_effect(tool_id: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG001
            if tool_id == "tool1":
                return mock_tool1
            return mock_tool2

        mock_get_tool.side_effect = get_tool_side_effect
        mock_tool_ctx = Mock()
        mock_get_tool_run_ctx.return_value = mock_tool_ctx

        result = await parallel_step.run(mock_run_data)

        assert result == ["tool1_result", "tool2_result"]
        assert mock_tool1._arun.call_count == 1
        assert mock_tool2._arun.call_count == 1


@pytest.mark.asyncio
async def test_parallel_step_run_with_failure() -> None:
    """Test ParallelStep run when one child step fails."""
    # Create mock steps where one fails
    step1 = Mock()
    step1.run = AsyncMock(return_value="result1")

    step2 = Mock()
    step2.run = AsyncMock(side_effect=RuntimeError("Step 2 failed"))

    step3 = Mock()

    # This step should be cancelled
    async def slow_step(run_data: Any) -> str:  # noqa: ANN401, ARG001
        await asyncio.sleep(10)
        return "result3"

    step3.run = slow_step

    parallel_step = ParallelStep(steps=[step1, step2, step3], step_name="parallel_execution")

    mock_run_data = Mock()

    with pytest.raises(RuntimeError, match="Step 2 failed"):
        await parallel_step.run(mock_run_data)

    # Verify step1 was called
    step1.run.assert_called_once_with(mock_run_data)
    step2.run.assert_called_once_with(mock_run_data)
    # Step3 should have been started but may have been cancelled


@pytest.mark.asyncio
async def test_parallel_step_run_empty_steps() -> None:
    """Test ParallelStep run with no child steps."""
    parallel_step = ParallelStep(steps=[], step_name="empty_parallel")

    mock_run_data = Mock()

    result = await parallel_step.run(mock_run_data)

    assert result == []


@pytest.mark.asyncio
async def test_parallel_step_run_single_step() -> None:
    """Test ParallelStep run with a single child step."""
    step1 = Mock()
    step1.run = AsyncMock(return_value="single_result")

    parallel_step = ParallelStep(steps=[step1], step_name="single_parallel")

    mock_run_data = Mock()

    result = await parallel_step.run(mock_run_data)

    assert result == ["single_result"]
    step1.run.assert_called_once_with(mock_run_data)


@pytest.mark.asyncio
async def test_parallel_step_run_preserves_order() -> None:
    """Test that ParallelStep preserves the order of outputs regardless of completion time."""

    # Create steps that complete in reverse order
    async def fast_step(run_data: Any) -> str:  # noqa: ANN401, ARG001
        await asyncio.sleep(0.001)
        return "fast"

    async def slow_step(run_data: Any) -> str:  # noqa: ANN401, ARG001
        await asyncio.sleep(0.005)
        return "slow"

    async def medium_step(run_data: Any) -> str:  # noqa: ANN401, ARG001
        await asyncio.sleep(0.003)
        return "medium"

    step1 = Mock()
    step1.run = slow_step

    step2 = Mock()
    step2.run = medium_step

    step3 = Mock()
    step3.run = fast_step

    parallel_step = ParallelStep(steps=[step1, step2, step3], step_name="ordered_parallel")

    mock_run_data = Mock()

    result = await parallel_step.run(mock_run_data)

    # Results should be in the order of steps, not completion order
    assert result == ["slow", "medium", "fast"]


def test_parallel_step_to_legacy_step() -> None:
    """Test ParallelStep to_legacy_step method."""
    step1 = InvokeToolStep(tool="tool1", step_name="step1")
    step2 = LLMStep(task="test task", step_name="step2")

    parallel_step = ParallelStep(steps=[step1, step2], step_name="parallel_execution")

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$parallel_execution_output"

    legacy_step = parallel_step.to_legacy_step(mock_plan)

    assert isinstance(legacy_step, PlanStep)
    assert "Execute 2 steps in parallel" in legacy_step.task
    assert legacy_step.output == "$parallel_execution_output"
    assert legacy_step.inputs == []


@pytest.mark.asyncio
async def test_parallel_step_with_local_data_value_outputs() -> None:
    """Test ParallelStep run when child steps return LocalDataValue."""
    step1 = Mock()
    step1.run = AsyncMock(return_value=LocalDataValue(value="value1", summary="Summary 1"))

    step2 = Mock()
    step2.run = AsyncMock(return_value=LocalDataValue(value="value2", summary="Summary 2"))

    parallel_step = ParallelStep(steps=[step1, step2], step_name="parallel_with_local_data")

    mock_run_data = Mock()

    result = await parallel_step.run(mock_run_data)

    assert len(result) == 2
    assert isinstance(result[0], LocalDataValue)
    assert result[0].value == "value1"
    assert result[0].summary == "Summary 1"
    assert isinstance(result[1], LocalDataValue)
    assert result[1].value == "value2"
    assert result[1].summary == "Summary 2"


def test_parallel_step_with_conditional_and_loop_blocks() -> None:
    """Test ParallelStep initialization with conditional and loop blocks."""
    step1 = InvokeToolStep(tool="tool1", step_name="step1")
    step2 = InvokeToolStep(tool="tool2", step_name="step2")

    conditional_block = ConditionalBlock(
        clause_step_indexes=[0, 5],
        parent_conditional_block=None,
    )

    loop_block = LoopBlock(
        start_step_index=0,
        end_step_index=10,
    )

    parallel_step = ParallelStep(
        steps=[step1, step2],
        step_name="parallel_in_control_flow",
        conditional_block=conditional_block,
        loop_block=loop_block,
    )

    assert parallel_step.conditional_block is conditional_block
    assert parallel_step.loop_block is loop_block
