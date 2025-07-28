"""Tests for the async functionality of the Tool class."""

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel, Field

from portia.clarification import InputClarification
from portia.errors import ToolHardError, ToolSoftError
from portia.execution_agents.output import LocalDataValue
from portia.tool import Tool, ToolRunContext
from tests.utils import (
    AdditionTool,
    ClarificationTool,
    ErrorTool,
    get_test_tool_context,
)


class AdditionToolSchema(BaseModel):
    """Input for AdditionTool."""

    a: int = Field(..., description="The first number to add")
    b: int = Field(..., description="The second number to add")


class AsyncAdditionTool(Tool):
    """Async version of AdditionTool for testing."""

    id: str = "add_tool"
    name: str = "Add Tool"
    description: str = "Use this tool to add two numbers together, it takes two numbers a + b"
    args_schema: type[BaseModel] = AdditionToolSchema
    output_schema: tuple[str, str] = ("int", "int: The value of the addition")

    def run(self, _: ToolRunContext, a: int, b: int) -> int:
        """Add the numbers."""
        return a + b

    async def arun(
        self,
        ctx: ToolRunContext,  # noqa: ARG002
        a: int,
        b: int,
    ) -> int:
        """Async add the numbers."""
        return a + b


class AsyncClarificationTool(ClarificationTool):
    """Async version of ClarificationTool for testing."""

    async def arun(
        self,
        ctx: ToolRunContext,
        user_guidance: str,
    ) -> InputClarification | None:
        """Async return a clarification."""
        if user_guidance == "test":
            return InputClarification(
                plan_run_id=ctx.plan_run.id,
                argument_name="test_arg",
                user_guidance=user_guidance,
                source="test",
            )
        return None


class AsyncErrorTool(ErrorTool):
    """Async version of ErrorTool for testing."""

    async def arun(
        self,
        _: ToolRunContext,
        error_str: str,
        return_uncaught_error: bool,
        return_soft_error: bool,
    ) -> None:
        """Async return an error."""
        if return_uncaught_error:
            raise ValueError(error_str)
        if return_soft_error:
            raise ToolSoftError(error_str)
        raise ToolHardError(error_str)


class SyncOnlyTool(AdditionTool):
    """Tool that only implements sync run method."""

    # Override arun to raise NotImplementedError to test fallback
    async def arun(
        self,
        ctx: ToolRunContext,
        *args: Any,
        **kwargs: Any,
    ) -> Any:  # noqa: ANN401
        """Raise NotImplementedError to test fallback."""
        raise NotImplementedError("Async run is not implemented")


@pytest.fixture
def async_add_tool() -> AsyncAdditionTool:
    """Fixture to create an async addition tool instance."""
    return AsyncAdditionTool()


@pytest.fixture
def async_clarification_tool() -> AsyncClarificationTool:
    """Fixture to create an async clarification tool instance."""
    return AsyncClarificationTool()


@pytest.fixture
def async_error_tool() -> AsyncErrorTool:
    """Fixture to create an async error tool instance."""
    return AsyncErrorTool()


@pytest.fixture
def sync_only_tool() -> SyncOnlyTool:
    """Fixture to create a sync-only tool instance."""
    return SyncOnlyTool()


@pytest.fixture
def tool_context() -> ToolRunContext:
    """Fixture to create a tool context."""
    return get_test_tool_context()


# Test async tool run functionality
@pytest.mark.asyncio
async def test_async_arun_success(
    async_add_tool: AsyncAdditionTool, tool_context: ToolRunContext
) -> None:
    """Test successful async run."""
    result = await async_add_tool.arun(tool_context, 5, 3)
    assert result == 8


@pytest.mark.asyncio
async def test_async_arun_clarification(
    async_clarification_tool: AsyncClarificationTool, tool_context: ToolRunContext
) -> None:
    """Test async run returning clarification."""
    result = await async_clarification_tool.arun(tool_context, "test")
    assert isinstance(result, InputClarification)
    assert result.argument_name == "test_arg"
    assert result.user_guidance == "test"


@pytest.mark.asyncio
async def test_async_arun_no_clarification(
    async_clarification_tool: AsyncClarificationTool, tool_context: ToolRunContext
) -> None:
    """Test async run returning None (no clarification)."""
    result = await async_clarification_tool.arun(tool_context, "no_clarification")
    assert result is None


@pytest.mark.asyncio
async def test_async_arun_tool_soft_error(
    async_error_tool: AsyncErrorTool, tool_context: ToolRunContext
) -> None:
    """Test async run raising ToolSoftError."""
    with pytest.raises(ToolSoftError, match="test error"):
        await async_error_tool.arun(tool_context, "test error", False, True)  # noqa: FBT003


@pytest.mark.asyncio
async def test_async_arun_tool_hard_error(
    async_error_tool: AsyncErrorTool, tool_context: ToolRunContext
) -> None:
    """Test async run raising ToolHardError."""
    with pytest.raises(ToolHardError, match="test error"):
        await async_error_tool.arun(tool_context, "test error", False, False)  # noqa: FBT003


@pytest.mark.asyncio
async def test_async_arun_uncaught_error(
    async_error_tool: AsyncErrorTool, tool_context: ToolRunContext
) -> None:
    """Test async run raising uncaught error (should be wrapped as ToolSoftError)."""
    with pytest.raises(ToolSoftError):
        await async_error_tool._arun(tool_context, "test error", True, False)  # noqa: FBT003


# Test internal async run methods (_arun, _arun_with_artifacts)
@pytest.mark.asyncio
async def test_async_internal_run_success(
    async_add_tool: AsyncAdditionTool, tool_context: ToolRunContext
) -> None:
    """Test successful internal async run."""
    result = await async_add_tool._arun(tool_context, 5, 3)
    assert isinstance(result, LocalDataValue)
    assert result.get_value() == 8


@pytest.mark.asyncio
async def test_async_internal_run_clarification(
    async_clarification_tool: AsyncClarificationTool, tool_context: ToolRunContext
) -> None:
    """Test internal async run returning clarification."""
    result = await async_clarification_tool._arun(tool_context, "test")
    assert isinstance(result, LocalDataValue)
    clarification = result.get_value()
    assert isinstance(clarification, list)
    assert len(clarification) == 1
    assert isinstance(clarification[0], InputClarification)


@pytest.mark.asyncio
async def test_async_internal_run_with_artifacts(
    async_add_tool: AsyncAdditionTool, tool_context: ToolRunContext
) -> None:
    """Test async run with artifacts."""
    content, artifact = await async_add_tool._arun_with_artifacts(tool_context, 5, 3)
    assert content == 8  # get_value() returns the actual value, not string
    assert isinstance(artifact, LocalDataValue)
    assert artifact.get_value() == 8


@pytest.mark.asyncio
async def test_async_internal_run_with_artifacts_clarification(
    async_clarification_tool: AsyncClarificationTool,
    tool_context: ToolRunContext,
) -> None:
    """Test async run with artifacts returning clarification."""
    content, artifact = await async_clarification_tool._arun_with_artifacts(tool_context, "test")
    # get_value() returns the actual list, not string representation
    assert isinstance(content, list)
    assert len(content) == 1
    assert isinstance(artifact, LocalDataValue)
    clarification = artifact.get_value()
    assert isinstance(clarification, list)
    assert len(clarification) == 1


# Test that sync-only tools fall back to sync implementation when async is not implemented
@pytest.mark.asyncio
async def test_sync_only_tool_arun_fallback(
    sync_only_tool: SyncOnlyTool, tool_context: ToolRunContext
) -> None:
    """Test that sync-only tool falls back to sync run when arun is not implemented."""
    # The fallback only works in _arun, not in direct arun
    with pytest.raises(NotImplementedError, match="Async run is not implemented"):
        await sync_only_tool.arun(tool_context, 5, 3)


@pytest.mark.asyncio
async def test_sync_only_tool_internal_arun_fallback(
    sync_only_tool: SyncOnlyTool, tool_context: ToolRunContext
) -> None:
    """Test that sync-only tool falls back to sync run in internal async method."""
    result = await sync_only_tool._arun(tool_context, 5, 3)
    assert isinstance(result, LocalDataValue)
    assert result.get_value() == 8


@pytest.mark.asyncio
async def test_sync_only_tool_internal_arun_with_artifacts_fallback(
    sync_only_tool: SyncOnlyTool,
    tool_context: ToolRunContext,
) -> None:
    """Test that sync-only tool falls back to sync run in internal async artifacts method."""
    content, artifact = await sync_only_tool._arun_with_artifacts(tool_context, 5, 3)
    assert content == 8  # get_value() returns the actual value, not string
    assert isinstance(artifact, LocalDataValue)
    assert artifact.get_value() == 8


# Test error handling in async tool methods
@pytest.mark.asyncio
async def test_async_internal_run_tool_soft_error(
    async_error_tool: AsyncErrorTool, tool_context: ToolRunContext
) -> None:
    """Test internal async run with ToolSoftError."""
    with pytest.raises(ToolSoftError, match="test error"):
        await async_error_tool._arun(tool_context, "test error", False, True)  # noqa: FBT003


@pytest.mark.asyncio
async def test_async_internal_run_tool_hard_error(
    async_error_tool: AsyncErrorTool, tool_context: ToolRunContext
) -> None:
    """Test internal async run with ToolHardError."""
    with pytest.raises(ToolHardError, match="test error"):
        await async_error_tool._arun(tool_context, "test error", False, False)  # noqa: FBT003


@pytest.mark.asyncio
async def test_async_internal_run_uncaught_error_wrapped(
    async_error_tool: AsyncErrorTool, tool_context: ToolRunContext
) -> None:
    """Test internal async run with uncaught error (should be wrapped as ToolSoftError)."""
    with pytest.raises(ToolSoftError):
        await async_error_tool._arun(tool_context, "test error", True, False)  # noqa: FBT003


@pytest.mark.asyncio
async def test_async_internal_run_with_artifacts_error(
    async_error_tool: AsyncErrorTool, tool_context: ToolRunContext
) -> None:
    """Test async run with artifacts when error occurs."""
    with pytest.raises(ToolSoftError):
        await async_error_tool._arun_with_artifacts(tool_context, "test error", True, False)  # noqa: FBT003


# Test async tool integration with LangChain
def test_to_langchain_async(
    async_add_tool: AsyncAdditionTool, tool_context: ToolRunContext
) -> None:
    """Test creating async LangChain tool."""
    langchain_tool = async_add_tool.to_langchain(tool_context, sync=False)
    assert langchain_tool.name == "Add_Tool"
    assert langchain_tool.coroutine is not None
    assert langchain_tool.func is None


def test_to_langchain_sync(async_add_tool: AsyncAdditionTool, tool_context: ToolRunContext) -> None:
    """Test creating sync LangChain tool from async tool."""
    langchain_tool = async_add_tool.to_langchain(tool_context, sync=True)
    assert langchain_tool.name == "Add_Tool"
    assert langchain_tool.func is not None
    assert langchain_tool.coroutine is None


def test_to_langchain_with_artifact_async(
    async_add_tool: AsyncAdditionTool, tool_context: ToolRunContext
) -> None:
    """Test creating async LangChain tool with artifacts."""
    langchain_tool = async_add_tool.to_langchain_with_artifact(tool_context, sync=False)
    assert langchain_tool.name == "Add_Tool"
    assert langchain_tool.coroutine is not None
    assert langchain_tool.func is None


def test_to_langchain_with_artifact_sync(
    async_add_tool: AsyncAdditionTool, tool_context: ToolRunContext
) -> None:
    """Test creating sync LangChain tool with artifacts from async tool."""
    langchain_tool = async_add_tool.to_langchain_with_artifact(tool_context, sync=True)
    assert langchain_tool.name == "Add_Tool"
    assert langchain_tool.func is not None
    assert langchain_tool.coroutine is None


@pytest.mark.asyncio
async def test_langchain_async_tool_execution(
    async_add_tool: AsyncAdditionTool, tool_context: ToolRunContext
) -> None:
    """Test executing async LangChain tool."""
    langchain_tool = async_add_tool.to_langchain(tool_context, sync=False)
    assert langchain_tool.coroutine is not None
    result = await langchain_tool.coroutine(5, 3)
    assert isinstance(result, LocalDataValue)
    assert result.get_value() == 8


@pytest.mark.asyncio
async def test_langchain_async_tool_with_artifacts_execution(
    async_add_tool: AsyncAdditionTool,
    tool_context: ToolRunContext,
) -> None:
    """Test executing async LangChain tool with artifacts."""
    langchain_tool = async_add_tool.to_langchain_with_artifact(tool_context, sync=False)
    assert langchain_tool.coroutine is not None
    content, artifact = await langchain_tool.coroutine(5, 3)
    assert content == 8  # get_value() returns the actual value, not string
    assert isinstance(artifact, LocalDataValue)
    assert artifact.get_value() == 8


class AdditionOutput(BaseModel):
    """Output schema for addition."""

    result: int


class StructuredAsyncAdditionTool(Tool):
    """Async addition tool with structured output."""

    id: str = "structured_async_addition_tool"
    name: str = "Structured Async Addition Tool"
    description: str = "A tool that adds two numbers together and returns a structured output"
    args_schema: type[BaseModel] = AdditionToolSchema
    output_schema: tuple[str, str] = (
        "AdditionOutput",
        "AdditionOutput: The result of the addition",
    )

    structured_output_schema: type[BaseModel] | None = AdditionOutput

    def run(self, _: ToolRunContext, a: int, b: int) -> AdditionOutput:
        """Sync add the numbers with structured output."""
        return AdditionOutput(result=a + b)

    async def arun(
        self,
        ctx: ToolRunContext,  # noqa: ARG002
        a: int,
        b: int,
    ) -> AdditionOutput:
        """Async add the numbers with structured output."""
        return AdditionOutput(result=a + b)


@pytest.fixture
def structured_async_tool() -> StructuredAsyncAdditionTool:
    """Fixture for structured async addition tool."""
    return StructuredAsyncAdditionTool()


# Test async tools with structured output schemas
@pytest.mark.asyncio
async def test_async_structured_output_success(
    structured_async_tool: StructuredAsyncAdditionTool, tool_context: ToolRunContext
) -> None:
    """Test async tool with structured output."""
    result = await structured_async_tool._arun(tool_context, 5, 3)
    assert isinstance(result, LocalDataValue)
    output = result.get_value()
    assert isinstance(output, AdditionOutput)
    assert output.result == 8


@pytest.mark.asyncio
async def test_async_structured_output_coercion(tool_context: ToolRunContext) -> None:
    """Test async tool with structured output coercion."""

    # Create a new tool instance with different arun method that returns a dict
    class CoercionTestTool(Tool):
        id: str = "coercion_test_tool"
        name: str = "Coercion Test Tool"
        description: str = "A tool that returns a dict"
        args_schema: type[BaseModel] = AdditionToolSchema
        output_schema: tuple[str, str] = ("dict", "dict: The result of the addition")
        structured_output_schema: type[BaseModel] | None = AdditionOutput

        def run(self, _: ToolRunContext, a: int, b: int) -> dict[str, int]:
            return {"result": a + b}

        async def arun(self, _: ToolRunContext, a: int, b: int) -> dict[str, int]:
            return {"result": a + b}

    test_tool = CoercionTestTool()
    result = await test_tool._arun(tool_context, 5, 3)
    assert isinstance(result, LocalDataValue)
    output = result.get_value()
    # The structured output schema should coerce the dict to AdditionOutput
    assert isinstance(output, AdditionOutput)
    assert output.result == 8


# Test edge cases for async tools
@pytest.mark.asyncio
async def test_async_tool_with_no_args(tool_context: ToolRunContext) -> None:
    """Test async tool with no arguments."""

    class NoArgsTool(Tool):
        id: str = "no_args_tool"
        name: str = "No Args Tool"
        description: str = "A tool with no arguments"
        output_schema: tuple[str, str] = ("str", "A simple string")

        def run(self, _: ToolRunContext) -> str:
            return "success"

        async def arun(self, _: ToolRunContext) -> str:
            return "success"

    tool = NoArgsTool()
    result = await tool._arun(tool_context)
    assert isinstance(result, LocalDataValue)
    assert result.get_value() == "success"


@pytest.mark.asyncio
async def test_async_tool_with_complex_args(tool_context: ToolRunContext) -> None:
    """Test async tool with complex arguments."""

    class ComplexArgsSchema(BaseModel):
        data: dict[str, list[int]]
        flag: bool = False

    class ComplexArgsTool(Tool):
        id: str = "complex_args_tool"
        name: str = "Complex Args Tool"
        description: str = "A tool with complex arguments"
        args_schema: type[BaseModel] = ComplexArgsSchema
        output_schema: tuple[str, str] = ("dict", "The processed data")

        def run(
            self, _: ToolRunContext, data: dict[str, list[int]], flag: bool = False
        ) -> dict[str, list[int]]:
            if flag:
                return {k: [x * 2 for x in v] for k, v in data.items()}
            return data

        async def arun(
            self, _: ToolRunContext, data: dict[str, list[int]], flag: bool = False
        ) -> dict[str, list[int]]:
            if flag:
                return {k: [x * 2 for x in v] for k, v in data.items()}
            return data

    tool = ComplexArgsTool()
    test_data = {"a": [1, 2, 3], "b": [4, 5, 6]}

    # Test without flag
    result = await tool._arun(tool_context, test_data, False)  # noqa: FBT003
    assert isinstance(result, LocalDataValue)
    assert result.get_value() == test_data

    # Test with flag
    result = await tool._arun(tool_context, test_data, True)  # noqa: FBT003
    assert isinstance(result, LocalDataValue)
    assert result.get_value() == {"a": [2, 4, 6], "b": [8, 10, 12]}


@pytest.mark.asyncio
async def test_async_tool_concurrent_execution(
    async_add_tool: AsyncAdditionTool, tool_context: ToolRunContext
) -> None:
    """Test concurrent execution of async tools."""
    tasks = [async_add_tool._arun(tool_context, i, i + 1) for i in range(5)]
    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        assert isinstance(result, LocalDataValue)
        assert result.get_value() == i + (i + 1)


# Integration tests for async tools
@pytest.mark.asyncio
async def test_async_tool_full_workflow(
    async_add_tool: AsyncAdditionTool, tool_context: ToolRunContext
) -> None:
    """Test complete async tool workflow."""
    # Test direct arun
    direct_result = await async_add_tool.arun(tool_context, 5, 3)
    assert direct_result == 8

    # Test internal _arun
    internal_result = await async_add_tool._arun(tool_context, 5, 3)
    assert isinstance(internal_result, LocalDataValue)
    assert internal_result.get_value() == 8

    # Test with artifacts
    content, artifact = await async_add_tool._arun_with_artifacts(tool_context, 5, 3)
    assert content == 8  # get_value() returns the actual value, not string
    assert isinstance(artifact, LocalDataValue)
    assert artifact.get_value() == 8

    # Test LangChain integration
    langchain_tool = async_add_tool.to_langchain(tool_context, sync=False)
    assert langchain_tool.coroutine is not None
    langchain_result = await langchain_tool.coroutine(5, 3)
    assert isinstance(langchain_result, LocalDataValue)
    assert langchain_result.get_value() == 8


@pytest.mark.asyncio
async def test_sync_fallback_full_workflow(
    sync_only_tool: SyncOnlyTool, tool_context: ToolRunContext
) -> None:
    """Test complete sync fallback workflow."""
    # Test direct arun (should raise NotImplementedError)
    with pytest.raises(NotImplementedError, match="Async run is not implemented"):
        await sync_only_tool.arun(tool_context, 5, 3)

    # Test internal _arun (should fall back to sync)
    internal_result = await sync_only_tool._arun(tool_context, 5, 3)
    assert isinstance(internal_result, LocalDataValue)
    assert internal_result.get_value() == 8

    # Test with artifacts (should fall back to sync)
    content, artifact = await sync_only_tool._arun_with_artifacts(tool_context, 5, 3)
    assert content == 8  # get_value() returns the actual value, not string
    assert isinstance(artifact, LocalDataValue)
    assert artifact.get_value() == 8

    # Test LangChain integration (should fall back to sync)
    langchain_tool = sync_only_tool.to_langchain(tool_context, sync=False)
    assert langchain_tool.coroutine is not None
    langchain_result = await langchain_tool.coroutine(5, 3)
    assert isinstance(langchain_result, LocalDataValue)
    assert langchain_result.get_value() == 8
