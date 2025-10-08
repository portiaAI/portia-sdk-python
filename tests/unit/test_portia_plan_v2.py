"""Test the Portia class with PlanV2."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field, HttpUrl, SecretStr

from portia.builder.exit import ExitStepResult
from portia.builder.plan_builder_v2 import PlanBuilderV2
from portia.builder.reference import Input
from portia.clarification import ActionClarification, ClarificationCategory
from portia.clarification_handler import ClarificationHandler
from portia.config import Config, StorageClass
from portia.errors import InvalidPlanRunStateError
from portia.execution_hooks import BeforeStepExecutionOutcome, ExecutionHooks
from portia.plan import Plan, PlanContext, PlanInput, Step
from portia.plan_run import PlanRun, PlanRunState
from portia.portia import Portia
from portia.tool import ReadyResponse, Tool, ToolRunContext, _ArgsSchemaPlaceholder
from portia.tool_registry import ToolRegistry
from tests.utils import AdditionTool, ClarificationTool

if TYPE_CHECKING:
    from collections.abc import Callable

    from portia.builder.plan_v2 import PlanV2
    from portia.clarification import Clarification
    from portia.execution_agents.output import Output


class ErrorClarificationHandler(ClarificationHandler):
    """A clarification handler that always triggers the error callback."""

    def handle(
        self,
        clarification: Clarification,
        on_resolution: Callable[[Clarification, object], None],  # noqa: ARG002
        on_error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle clarification by triggering error callback."""
        on_error(clarification, "Test error message")


class SuccessClarificationHandler(ClarificationHandler):
    """A clarification handler that always triggers the success callback."""

    def handle(
        self,
        clarification: Clarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002
    ) -> None:
        """Handle clarification by triggering success callback."""
        on_resolution(clarification, "Test success message")


class ReadyTestTool(Tool):
    """A tool that can simulate readiness states for testing."""

    id: str = "ready_test_tool"
    name: str = "Ready Test Tool"
    description: str = "A tool for testing readiness flow"
    args_schema: type[BaseModel] = _ArgsSchemaPlaceholder
    output_schema: tuple[str, str] = ("str", "Test output from ready tool")

    _is_ready: bool = False

    def ready(self, ctx: ToolRunContext) -> ReadyResponse:
        """Check if tool is ready."""
        if self._is_ready:
            return ReadyResponse(ready=True, clarifications=[])

        clarification = ActionClarification(
            user_guidance="Please authenticate with the tool",
            plan_run_id=ctx.plan_run.id,
            action_url=HttpUrl("https://example.com/auth"),
            category=ClarificationCategory.ACTION,
        )
        return ReadyResponse(ready=False, clarifications=[clarification])

    def run(self, ctx: ToolRunContext) -> str:  # noqa: ARG002
        """Run the tool."""
        return "Tool executed successfully"

    def mark_ready(self) -> None:
        """Mark the tool as ready (simulate completing authentication)."""
        self._is_ready = True


class ErrorTool(Tool):
    """A tool that always raises an error for testing."""

    id: str = "error_tool"
    name: str = "Error Tool"
    description: str = "A tool that always raises an error"
    args_schema: type[BaseModel] = _ArgsSchemaPlaceholder
    output_schema: tuple[str, str] = ("str", "Error output")

    def run(self, ctx: ToolRunContext) -> str:  # noqa: ARG002
        """Run the tool."""
        raise RuntimeError("Tool execution failed")


def _build_addition_plan() -> PlanV2:
    """Build a simple addition plan using the builder."""
    return (
        PlanBuilderV2("Add two numbers")
        .input(name="num_a", description="First number to add")
        .input(name="num_b", description="Second number to add")
        .invoke_tool_step(
            step_name="Add numbers",
            tool="add_tool",
            args={"a": Input("num_a"), "b": Input("num_b")},
        )
        .build()
    )


def _create_portia_with_storage(storage_type: str, tmp_dir: str) -> Portia:
    """Create a Portia instance with the specified storage type."""
    if storage_type == "disk":
        config = Config.from_default(
            storage_class=StorageClass.DISK,
            storage_dir=tmp_dir,
            openai_api_key=SecretStr("123"),
        )
    else:
        config = Config.from_default(openai_api_key=SecretStr("123"))
    tool_registry = ToolRegistry([AdditionTool(), ClarificationTool()])
    return Portia(config=config, tools=tool_registry)


@pytest.mark.parametrize("storage_type", ["local", "disk"])
def test_portia_run_plan_v2_sync_mainline(storage_type: str) -> None:
    """Test running a PlanV2 synchronously works as expected for mainline case."""
    plan = (
        PlanBuilderV2("Test sync mainline")
        .function_step(function=lambda: "Step 1 result")
        .function_step(function=lambda: "Step 2 result")
        .build()
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        portia = _create_portia_with_storage(storage_type, tmp_dir)
        plan_run = portia.run_plan(plan)

        assert plan_run.state == PlanRunState.COMPLETE
        assert plan_run.plan_id == plan.id
        assert plan_run.outputs.final_output is not None
        assert plan_run.outputs.final_output.get_value() == "Step 2 result"
        assert plan_run.outputs.step_outputs["$step_0_output"].get_value() == "Step 1 result"
        assert plan_run.outputs.step_outputs["$step_1_output"].get_value() == "Step 2 result"

        if storage_type == "disk" and tmp_dir:
            plan_files = list(Path(tmp_dir).glob("plan-*.json"))
            run_files = list(Path(tmp_dir).glob("prun-*.json"))
            assert len(plan_files) == 1
            assert len(run_files) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_type", ["local", "disk"])
async def test_portia_run_plan_v2_async_mainline(storage_type: str) -> None:
    """Test running a PlanV2 asynchronously works as expected for mainline case."""
    plan = (
        PlanBuilderV2("Test async mainline")
        .function_step(function=lambda: "Step 1 result")
        .function_step(function=lambda: "Step 2 result")
        .build()
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        portia = _create_portia_with_storage(storage_type, tmp_dir)
        end_user = await portia.ainitialize_end_user()
        plan_run = await portia.arun_plan(plan, end_user)

        assert plan_run.state == PlanRunState.COMPLETE
        assert plan_run.plan_id == plan.id
        assert plan_run.outputs.final_output is not None
        assert plan_run.outputs.final_output.get_value() == "Step 2 result"
        assert plan_run.outputs.step_outputs["$step_0_output"].get_value() == "Step 1 result"
        assert plan_run.outputs.step_outputs["$step_1_output"].get_value() == "Step 2 result"

        if storage_type == "disk" and tmp_dir:
            # Verify files were created for disk storage
            plan_files = list(Path(tmp_dir).glob("plan-*.json"))
            run_files = list(Path(tmp_dir).glob("prun-*.json"))
            assert len(plan_files) == 1
            assert len(run_files) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plan_run_inputs",
    [
        [
            PlanInput(name="num_a", description="First number to add", value=10),
            PlanInput(name="num_b", description="Second number to add", value=7),
        ],
        [
            {"name": "num_a", "description": "First number to add", "value": 10},
            {"name": "num_b", "description": "Second number to add", "value": 7},
        ],
        {
            "num_a": 10,
            "num_b": 7,
        },
        {
            # Additional inputs should be ignored
            "num_a": 10,
            "num_b": 7,
            "num_c": 10,
        },
        [
            {"incorrect_key": "num_a", "error": "Error"},
        ],
        "error",
    ],
)
async def test_portia_run_plan_v2_with_plan_inputs(
    portia: Portia,
    plan_run_inputs: list[PlanInput] | list[dict[str, object]] | dict[str, object],
) -> None:
    """Test running a PlanV2 asynchronously with plan inputs in different formats."""
    plan = _build_addition_plan()
    end_user = portia.initialize_end_user()

    if plan_run_inputs == "error" or (
        isinstance(plan_run_inputs, list)
        and isinstance(plan_run_inputs[0], dict)
        and "error" in plan_run_inputs[0]
    ):
        with pytest.raises(ValueError):  # noqa: PT011
            await portia.arun_plan(plan, end_user, plan_run_inputs=plan_run_inputs)
        return

    plan_run = await portia.arun_plan(plan, end_user, plan_run_inputs=plan_run_inputs)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.plan_id == plan.id
    assert len(plan_run.plan_run_inputs) == 2
    assert plan_run.plan_run_inputs["num_a"].get_value() == 10
    assert plan_run.plan_run_inputs["num_b"].get_value() == 7
    assert plan_run.outputs.final_output is not None
    assert plan_run.outputs.final_output.get_value() == 17


def _build_plan_with_default_inputs() -> PlanV2:
    """Build a plan with some inputs that have defaults and some that don't."""
    return (
        PlanBuilderV2("Test plan with default inputs")
        .input(name="required_input", description="This input is required")
        .input(
            name="optional_input",
            description="This input has a default",
            default_value="default_value",
        )
        .function_step(
            function=lambda x, y: f"{x}_{y}",
            args={"x": Input("required_input"), "y": Input("optional_input")},
        )
        .build()
    )


@pytest.mark.parametrize(
    ("plan_run_inputs", "should_succeed", "expected_output"),
    [
        # Case 1: Missing required input - should fail
        (
            {"optional_input": "provided_optional"},
            False,
            None,
        ),
        # Case 2: Missing optional input (has default) - should succeed
        (
            {"required_input": "provided_required"},
            True,
            "provided_required_default_value",
        ),
        # Case 3: Both inputs provided - should succeed
        (
            {
                "required_input": "provided_required",
                "optional_input": "provided_optional",
            },
            True,
            "provided_required_provided_optional",
        ),
    ],
)
def test_portia_run_plan_v2_missing_inputs(
    portia: Portia,
    plan_run_inputs: dict[str, object],
    should_succeed: bool,
    expected_output: str | None,
) -> None:
    """Test running a PlanV2 asynchronously with missing inputs.

    Should work with defaults, fail without.
    """
    plan = _build_plan_with_default_inputs()
    end_user = portia.initialize_end_user()

    if not should_succeed:
        with pytest.raises(ValueError, match="Missing required plan input values"):
            asyncio.run(portia.arun_plan(plan, end_user, plan_run_inputs=plan_run_inputs))
        return

    plan_run = asyncio.run(portia.arun_plan(plan, end_user, plan_run_inputs=plan_run_inputs))

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.plan_id == plan.id
    assert plan_run.outputs.final_output is not None
    assert plan_run.outputs.final_output.get_value() == expected_output


def _build_plan_with_multiple_default_inputs() -> PlanV2:
    """Build a plan with multiple inputs, some with defaults and some without."""
    return (
        PlanBuilderV2("Test plan with multiple default inputs")
        .input(name="required_input_1", description="This input is required")
        .input(name="required_input_2", description="This input is also required")
        .input(
            name="optional_input_1",
            description="This input has a default",
            default_value="default_1",
        )
        .input(
            name="optional_input_2",
            description="This input also has a default",
            default_value="default_2",
        )
        .function_step(
            function=lambda a, b, c, d: f"{a}_{b}_{c}_{d}",
            args={
                "a": Input("required_input_1"),
                "b": Input("required_input_2"),
                "c": Input("optional_input_1"),
                "d": Input("optional_input_2"),
            },
        )
        .build()
    )


def _build_plan_with_all_default_inputs() -> PlanV2:
    """Build a plan where all inputs have default values."""
    return (
        PlanBuilderV2("Test plan with all default inputs")
        .input(
            name="input_1",
            description="This input has a default",
            default_value="default_1",
        )
        .input(
            name="input_2",
            description="This input also has a default",
            default_value="default_2",
        )
        .function_step(
            function=lambda x, y: f"{x}_{y}",
            args={"x": Input("input_1"), "y": Input("input_2")},
        )
        .build()
    )


@pytest.mark.parametrize(
    ("plan_run_inputs", "expected_output"),
    [
        # Case 1: Some required inputs provided, optional inputs use defaults
        (
            {"required_input_1": "provided_1", "required_input_2": "provided_2"},
            "provided_1_provided_2_default_1_default_2",
        ),
        # Case 2: Required inputs provided, some optional inputs provided, others use defaults
        (
            {
                "required_input_1": "provided_1",
                "required_input_2": "provided_2",
                "optional_input_1": "custom_1",
            },
            "provided_1_provided_2_custom_1_default_2",
        ),
        # Case 3: All inputs provided
        (
            {
                "required_input_1": "provided_1",
                "required_input_2": "provided_2",
                "optional_input_1": "custom_1",
                "optional_input_2": "custom_2",
            },
            "provided_1_provided_2_custom_1_custom_2",
        ),
    ],
)
def test_portia_run_plan_v2_with_default_inputs_comprehensive(
    portia: Portia,
    plan_run_inputs: dict[str, object],
    expected_output: str,
) -> None:
    """Test running a PlanV2 with various combinations of default and provided inputs."""
    plan = _build_plan_with_multiple_default_inputs()
    end_user = portia.initialize_end_user()

    plan_run = asyncio.run(portia.arun_plan(plan, end_user, plan_run_inputs=plan_run_inputs))

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.plan_id == plan.id
    assert plan_run.outputs.final_output is not None
    assert plan_run.outputs.final_output.get_value() == expected_output


@pytest.mark.parametrize(
    ("plan_run_inputs", "expected_output"),
    [
        # Case 1: No inputs provided, all use defaults
        ({}, "default_1_default_2"),
        # Case 2: Some inputs provided, others use defaults
        ({"input_1": "custom_1"}, "custom_1_default_2"),
        # Case 3: All inputs provided (overriding defaults)
        ({"input_1": "custom_1", "input_2": "custom_2"}, "custom_1_custom_2"),
    ],
)
def test_portia_run_plan_v2_with_all_default_inputs(
    portia: Portia,
    plan_run_inputs: dict[str, object],
    expected_output: str,
) -> None:
    """Test running a PlanV2 where all inputs have default values."""
    plan = _build_plan_with_all_default_inputs()
    end_user = portia.initialize_end_user()

    plan_run = asyncio.run(portia.arun_plan(plan, end_user, plan_run_inputs=plan_run_inputs))

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.plan_id == plan.id
    assert plan_run.outputs.final_output is not None
    assert plan_run.outputs.final_output.get_value() == expected_output


@pytest.mark.asyncio
async def test_run_builder_plan_execution_hooks(portia: Portia) -> None:
    """Test that execution hooks are called when running a PlanV2."""
    execution_hooks = ExecutionHooks(
        before_plan_run=MagicMock(),
        before_step_execution=MagicMock(),
        after_step_execution=MagicMock(),
        after_plan_run=MagicMock(),
    )
    portia.execution_hooks = execution_hooks
    plan = (
        PlanBuilderV2("Test execution hooks")
        .function_step(function=lambda: "Step 1 result")
        .function_step(function=lambda: "Step 2 result")
        .build()
    )

    plan_run = await portia.arun_plan(plan)

    assert plan_run.state == PlanRunState.COMPLETE
    assert execution_hooks.before_plan_run.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
    assert execution_hooks.before_step_execution.call_count == 2  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
    assert execution_hooks.after_step_execution.call_count == 2  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
    assert execution_hooks.after_plan_run.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_run_builder_plan_execution_hooks_with_skip(portia: Portia) -> None:
    """Test that before_step_execution can skip steps for PlanV2."""

    def before_step_execution(plan: Plan, plan_run: PlanRun, step: Step):  # noqa: ANN202, ARG001
        return (
            BeforeStepExecutionOutcome.SKIP
            if step.output == "$step_0_output"
            else BeforeStepExecutionOutcome.CONTINUE
        )

    execution_hooks = ExecutionHooks(
        before_step_execution=before_step_execution,
        after_step_execution=MagicMock(),
    )
    portia.execution_hooks = execution_hooks
    plan = (
        PlanBuilderV2("Test execution hooks with skip")
        .function_step(function=lambda: "Step 1 result")
        .function_step(function=lambda: "Step 2 result")
        .build()
    )

    plan_run = await portia.arun_plan(plan)

    assert plan_run.state == PlanRunState.COMPLETE
    assert execution_hooks.after_step_execution.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_run_builder_plan_execution_hooks_after_step_error(portia: Portia) -> None:
    """Test that an error in after_step_execution hook causes the plan to fail."""

    def after_step_execution(plan: Plan, plan_run: PlanRun, step: Step, output: Output):  # noqa: ANN202, ARG001
        raise RuntimeError("After step execution hook failed")

    execution_hooks = ExecutionHooks(
        after_step_execution=after_step_execution,
    )
    portia.execution_hooks = execution_hooks
    plan = (
        PlanBuilderV2("Test execution hooks after step error")
        .function_step(function=lambda: "Step 1 result")
        .build()
    )

    plan_run = await portia.arun_plan(plan)

    assert plan_run.state == PlanRunState.FAILED


@pytest.mark.asyncio
@patch("portia.portia.get_current_run_tree")
async def test_run_builder_plan_appends_plan_run_to_trace_metadata(
    mock_get_run_tree: MagicMock, portia: Portia
) -> None:
    """Ensure plan_run id is appended to trace metadata."""
    mock_run_tree = MagicMock()
    mock_get_run_tree.return_value = mock_run_tree

    plan = PlanBuilderV2("Trace metadata").function_step(function=lambda: "Step result").build()
    plan_run = await portia.arun_plan(plan)

    mock_run_tree.add_metadata.assert_called_once()
    metadata_arg = mock_run_tree.add_metadata.call_args.args[0]
    assert metadata_arg["plan_run_id"] == str(plan_run.id)


class FinalOutputSchema(BaseModel):
    """Schema for final output testing."""

    message: str = Field(description="The result value")
    success: bool = Field(description="The count value")
    # The summariser usually adds this automatically, but we add this here as we mock out the
    # summariser
    fo_summary: str | None = Field(default=None, description="Final output summary")


class OverriddenOutputSchema(BaseModel):
    """Schema for overridden final output testing."""

    message: str = Field(description="The message value")
    error: bool = Field(description="Error indicator")
    # The summariser usually adds this automatically, but we add this here as we mock out the
    # summariser
    fo_summary: str | None = Field(default=None, description="Final output summary")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("plan_schema", "runtime_schema", "expected_schema_type"),
    [
        # Case 1: Plan has schema, no runtime override
        (FinalOutputSchema, None, FinalOutputSchema),
        # Case 2: Plan has schema, runtime override - runtime takes precedence
        (FinalOutputSchema, OverriddenOutputSchema, OverriddenOutputSchema),
    ],
)
async def test_portia_run_plan_v2_with_final_output_schema(
    portia: Portia,
    plan_schema: type[BaseModel],
    runtime_schema: type[BaseModel] | None,
    expected_schema_type: type[BaseModel],
) -> None:
    """Test running a PlanV2 with final output schemas (async only)."""
    plan = (
        PlanBuilderV2("Test final output schema")
        .function_step(function=lambda: "test result")
        .final_output(output_schema=plan_schema)
    ).build()

    # Mock the summarizer to just always return "test result"
    mock_summarizer = mock.MagicMock()
    ret_val = (
        FinalOutputSchema(
            message="test result",
            success=True,
            fo_summary="test result",
        )
        if expected_schema_type == FinalOutputSchema
        else OverriddenOutputSchema(
            message="test result",
            error=True,
            fo_summary="test result",
        )
    )
    mock_summarizer.create_summary.return_value = ret_val

    with mock.patch(
        "portia.portia.FinalOutputSummarizer",
        return_value=mock_summarizer,
    ):
        end_user = await portia.ainitialize_end_user()
        plan_run = await portia.arun_plan(plan, end_user, structured_output_schema=runtime_schema)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.plan_id == plan.id
    assert plan_run.structured_output_schema == expected_schema_type
    assert plan_run.outputs.final_output is not None
    assert plan_run.outputs.final_output.get_value().message == "test result"  #  pyright: ignore[reportOptionalMemberAccess]
    assert mock_summarizer.create_summary.call_count == 1
    assert (
        mock_summarizer.create_summary.call_args.kwargs["plan_run"].structured_output_schema
        == expected_schema_type
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_output_schema", "plan_output_schema", "use_clarification_handler"),
    [
        # Case 1: No structured output, explicit clarification handling
        (None, None, False),
        # Case 2: Both tool and plan have structured output schemas
        (FinalOutputSchema, OverriddenOutputSchema, False),
        # Case 3: No structured output, clarification handler
        (None, None, True),
        # Case 4: Both plan has structured output schemas, clarification handler
        (FinalOutputSchema, OverriddenOutputSchema, True),
    ],
)
async def test_portia_run_plan_v2_with_clarification_error(
    tool_output_schema: type[BaseModel] | None,
    plan_output_schema: type[BaseModel] | None,
    use_clarification_handler: bool,
) -> None:
    """Test running a PlanV2 when a clarification error occurs."""
    # Build the plan with a clarification step instead of error tool
    plan_builder = PlanBuilderV2("Test clarification error")
    plan_builder = plan_builder.invoke_tool_step(
        step_name="Clarification step",
        tool="clarification_tool",
        args={"user_guidance": "This will trigger an error"},
        output_schema=tool_output_schema,
    )

    if plan_output_schema:
        plan_builder = plan_builder.final_output(output_schema=plan_output_schema)

    plan = plan_builder.build()

    tool_registry = ToolRegistry([AdditionTool(), ClarificationTool()])
    portia = Portia(
        config=Config.from_default(openai_api_key=SecretStr("123")), tools=tool_registry
    )

    if use_clarification_handler:
        portia.execution_hooks = ExecutionHooks(clarification_handler=ErrorClarificationHandler())

    end_user = await portia.ainitialize_end_user()

    plan_run = await portia.arun_plan(plan, end_user)
    if not use_clarification_handler:
        # Explicitly handle the clarification and resume
        assert plan_run.state == PlanRunState.NEED_CLARIFICATION
        outstanding_clarifications = plan_run.get_outstanding_clarifications()
        assert len(outstanding_clarifications) == 1
        clarification = outstanding_clarifications[0]
        portia.error_clarification(clarification, "Proceed to error", plan_run)
        plan_run = await portia.aresume(plan_run, plan=plan)

    assert plan_run.state == PlanRunState.FAILED
    assert plan_run.plan_id == plan.id
    assert plan_run.outputs.final_output is None

    # Check that the plan can't be resumed afterwards
    plan_run = await portia.aresume(plan_run, plan=plan)

    assert plan_run.state == PlanRunState.FAILED
    assert plan_run.plan_id == plan.id
    assert plan_run.outputs.final_output is None


@pytest.mark.asyncio
async def test_portia_resume_plan_v2_not_ready(portia: Portia) -> None:
    """Test resuming a PlanV2 when the plan isn't ready (NOT_STARTED state)."""
    plan = (
        PlanBuilderV2("Test resume not ready")
        .invoke_tool_step(
            step_name="Add step",
            tool="add_tool",
            args={"a": 10, "b": 5},
        )
        .build()
    )

    end_user = await portia.ainitialize_end_user()
    legacy_plan = plan.to_legacy_plan(
        PlanContext(
            query=plan.label,
            tool_ids=[tool.id for tool in portia.tool_registry.get_tools()],
        )
    )
    plan_run = await portia._aget_plan_run_from_plan(legacy_plan, end_user, None)
    plan_run.state = PlanRunState.NOT_STARTED

    # Resume should work from NOT_STARTED state
    result = await portia.aresume(plan_run, plan=plan)

    assert result.state == PlanRunState.COMPLETE
    assert result.outputs.final_output is not None
    assert result.outputs.final_output.get_value() == 15  # 10 + 5


@pytest.mark.parametrize("use_clarification_handler", [True, False])
def test_portia_resume_plan_v2_after_clarification_success(
    use_clarification_handler: bool,
) -> None:
    """Test resuming a PlanV2 after a clarification is resolved successfully."""
    plan = (
        PlanBuilderV2("Test resume after clarification")
        .invoke_tool_step(
            step_name="Clarification step",
            tool="clarification_tool",
            args={"user_guidance": "Please provide input"},
        )
        .function_step(function=lambda: "Final result")
        .build()
    )
    tool_registry = ToolRegistry([ClarificationTool()])
    portia = Portia(
        config=Config.from_default(openai_api_key=SecretStr("123")), tools=tool_registry
    )
    if use_clarification_handler:
        portia.execution_hooks = ExecutionHooks(clarification_handler=SuccessClarificationHandler())

    end_user = portia.initialize_end_user()
    plan_run = portia.run_plan(plan, end_user=end_user)

    if not use_clarification_handler:
        # Explicitly handle the clarification and resume
        assert plan_run.state == PlanRunState.NEED_CLARIFICATION
        outstanding_clarifications = plan_run.get_outstanding_clarifications()
        assert len(outstanding_clarifications) == 1
        clarification = outstanding_clarifications[0]
        assert clarification.user_guidance == "Please provide input"
        plan_run = portia.resolve_clarification(clarification, "Resolved input", plan_run)
        plan_run = portia.resume(plan_run, plan=plan)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output is not None
    assert plan_run.outputs.final_output.get_value() == "Final result"


@pytest.mark.asyncio
async def test_portia_resume_plan_v2_after_clarification_success_async(portia: Portia) -> None:
    """Test resuming a PlanV2 after a clarification is resolved successfully.

    Note: This test currently fails due to a bug in clarification handling where
    resolved clarifications don't properly advance to the next step on resume.
    The issue affects both sync and async modes equally.
    """
    plan = (
        PlanBuilderV2("Test resume after clarification")
        .invoke_tool_step(
            step_name="Clarification step",
            tool="clarification_tool",
            args={"user_guidance": "Please provide input"},
        )
        .function_step(function=lambda: "Final result")
        .build()
    )

    end_user = await portia.ainitialize_end_user()
    plan_run = await portia.arun_plan(plan, end_user)

    # Plan should be waiting for clarification
    assert plan_run.state == PlanRunState.NEED_CLARIFICATION
    outstanding_clarifications = plan_run.get_outstanding_clarifications()
    assert len(outstanding_clarifications) == 1
    clarification = outstanding_clarifications[0]
    assert clarification.user_guidance == "Please provide input"

    # Resolve the clarification
    plan_run = portia.resolve_clarification(clarification, "Resolved input", plan_run)

    # Resume the plan
    result = await portia.aresume(plan_run, plan=plan)

    assert result.state == PlanRunState.COMPLETE
    assert result.outputs.final_output is not None
    assert result.outputs.final_output.get_value() == "Final result"


def _make_local_portia() -> Portia:
    """Construct a Portia instance using local storage for unit tests."""
    config = Config.from_default(storage_class=StorageClass.MEMORY, openai_api_key=SecretStr("123"))
    return Portia(config=config)


@pytest.mark.parametrize("is_async", [False, True])
def test_exit_step_non_error_completes_and_skips_remaining(is_async: bool) -> None:
    """Plan with a non-error ExitStep should complete and skip subsequent steps."""
    plan = (
        PlanBuilderV2("Exit early")
        .function_step(function=lambda: "first")
        .exit(message="Stopping now")
        .function_step(function=lambda: "should not run")
        .build()
    )

    portia = _make_local_portia()

    if is_async:
        end_user = asyncio.run(portia.ainitialize_end_user())
        plan_run = asyncio.run(portia.arun_plan(plan, end_user))
    else:
        plan_run = portia.run_plan(plan)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output is not None
    final_value = plan_run.outputs.final_output.get_value()
    # Final output should carry the ExitStepResult from the exit step
    assert isinstance(final_value, ExitStepResult)
    assert final_value.message == "Stopping now"
    # Ensure the step after the exit did not run
    assert "$step_2_output" not in plan_run.outputs.step_outputs


@pytest.mark.parametrize("is_async", [False, True])
def test_exit_step_error_sets_failed_and_message(is_async: bool) -> None:
    """Plan with an error ExitStep should fail and expose the error message as final output."""
    plan = (
        PlanBuilderV2("Exit with error")
        .function_step(function=lambda: "ok")
        .exit(message="fatal problem occurred", error=True)
        .function_step(function=lambda: "should not run")
        .build()
    )

    portia = _make_local_portia()

    if is_async:
        end_user = asyncio.run(portia.ainitialize_end_user())
        plan_run = asyncio.run(portia.arun_plan(plan, end_user))
    else:
        plan_run = portia.run_plan(plan)

    assert plan_run.state == PlanRunState.FAILED
    assert plan_run.outputs.final_output is not None
    # Error handler converts to string and sets as final output
    assert plan_run.outputs.final_output.get_value() == "fatal problem occurred"
    # Ensure the step after the exit did not run
    assert "$step_2_output" not in plan_run.outputs.step_outputs


@pytest.mark.parametrize("is_async", [False, True])
def test_exit_step_message_reference_resolution_end_to_end(is_async: bool) -> None:
    """Exit message should resolve StepOutput references end-to-end through Portia execution."""
    plan = (
        PlanBuilderV2("Exit with reference")
        .function_step(function=lambda: "computed value")
        .exit(message="Processed {{ StepOutput(0) }} successfully")
        .build()
    )

    portia = _make_local_portia()

    if is_async:
        end_user = asyncio.run(portia.ainitialize_end_user())
        plan_run = asyncio.run(portia.arun_plan(plan, end_user))
    else:
        plan_run = portia.run_plan(plan)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output is not None
    final_value = plan_run.outputs.final_output.get_value()
    assert isinstance(final_value, ExitStepResult)
    assert final_value.message == "Processed computed value successfully"


@pytest.mark.asyncio
async def test_portia_resume_plan_v2_keyboard_interrupt(portia: Portia) -> None:
    """Test running a PlanV2 asynchronously handles keyboard interrupt."""
    plan = (
        PlanBuilderV2("Test keyboard interrupt async")
        .function_step(function=lambda: "Step result")
        .build()
    )
    end_user = await portia.ainitialize_end_user()

    # Create a plan run in progress state
    legacy_plan = plan.to_legacy_plan(
        PlanContext(
            query=plan.label,
            tool_ids=[tool.id for tool in portia.tool_registry.get_tools()],
        )
    )
    plan_run = await portia._aget_plan_run_from_plan(legacy_plan, end_user, None)
    plan_run.state = PlanRunState.IN_PROGRESS

    # Mock _execute_builder_plan to raise KeyboardInterrupt
    with mock.patch.object(portia, "_execute_builder_plan", side_effect=KeyboardInterrupt):
        await portia.aresume(plan_run=plan_run, plan=plan)

    # Should set the plan run state to FAILED when KeyboardInterrupt occurs
    assert plan_run.state == PlanRunState.FAILED


@pytest.mark.asyncio
async def test_portia_get_final_output_handles_summary_error(portia: Portia) -> None:
    """Test that final output is set even if summary generation fails with PlanV2."""
    # Create a single step plan with final output processing
    plan = (
        PlanBuilderV2("Test summary error handling")
        .function_step(function=lambda: "Some output")
        .final_output()  # Enable final output processing which includes summary generation
        .build()
    )

    # Mock the SummarizerAgent to raise an exception during summary generation
    mock_agent = mock.MagicMock()
    mock_agent.create_summary.side_effect = Exception("Summary failed")

    with mock.patch(
        "portia.portia.FinalOutputSummarizer",
        return_value=mock_agent,
    ):
        end_user = await portia.ainitialize_end_user()
        plan_run = await portia.arun_plan(plan, end_user)

        assert plan_run.state == PlanRunState.COMPLETE
        assert plan_run.plan_id == plan.id
        assert plan_run.outputs.final_output is not None
        assert plan_run.outputs.final_output.get_value() == "Some output"
        assert plan_run.outputs.final_output.get_summary() is None


@pytest.mark.asyncio
async def test_portia_wait_for_ready_max_retries_plan_v2(portia: Portia) -> None:
    """Test wait for ready with max retries using PlanV2."""
    plan = (
        PlanBuilderV2("Test max retries")
        .invoke_tool_step(
            step_name="Clarification step",
            tool="clarification_tool",
            args={"user_guidance": "Please provide input"},
        )
        .build()
    )

    legacy_plan = plan.to_legacy_plan(
        PlanContext(
            query=plan.label,
            tool_ids=[tool.id for tool in portia.tool_registry.get_tools()],
        )
    )

    end_user = await portia.ainitialize_end_user()
    plan_run = await portia._aget_plan_run_from_plan(legacy_plan, end_user, None)
    plan_run.state = PlanRunState.NEED_CLARIFICATION

    portia.storage.save_plan(legacy_plan)
    portia.storage.save_plan_run(plan_run)

    with pytest.raises(InvalidPlanRunStateError):
        portia.wait_for_ready(plan_run, max_retries=0)


@pytest.mark.asyncio
async def test_portia_wait_for_ready_backoff_period_plan_v2(portia: Portia) -> None:
    """Test wait for ready with backoff period using PlanV2."""
    plan = (
        PlanBuilderV2("Test backoff period")
        .invoke_tool_step(
            step_name="Clarification step",
            tool="clarification_tool",
            args={"user_guidance": "Please provide input"},
        )
        .build()
    )

    legacy_plan = plan.to_legacy_plan(
        PlanContext(
            query=plan.label,
            tool_ids=[tool.id for tool in portia.tool_registry.get_tools()],
        )
    )

    end_user = await portia.ainitialize_end_user()
    plan_run = await portia._aget_plan_run_from_plan(legacy_plan, end_user, None)
    plan_run.state = PlanRunState.NEED_CLARIFICATION

    portia.storage.save_plan(legacy_plan)
    portia.storage.get_plan_run = mock.MagicMock(return_value=plan_run)

    with mock.patch.object(portia, "_check_remaining_tool_readiness") as mock_check:
        mock_check.return_value = [MagicMock()]
        with pytest.raises(InvalidPlanRunStateError):
            portia.wait_for_ready(plan_run, max_retries=1, backoff_start_time_seconds=0)


@pytest.mark.asyncio
async def test_portia_plan_v2_initial_readiness_check_with_action_clarification() -> None:
    """Test PlanV2 run_plan flow with initial tool readiness check requiring clarification."""
    ready_tool = ReadyTestTool()

    config = Config.from_default(
        openai_api_key=SecretStr("123"),
    )
    tool_registry = ToolRegistry([ready_tool, ClarificationTool()])
    portia = Portia(config=config, tools=tool_registry)

    plan = (
        PlanBuilderV2("Test readiness check with action clarification")
        .invoke_tool_step(
            step_name="Use ready tool",
            tool="ready_test_tool",
            args={},
        )
        .build()
    )

    # Run the plan - should stop due to tool not being ready
    end_user = await portia.ainitialize_end_user()
    plan_run = await portia.arun_plan(plan, end_user)

    # Verify plan is waiting for clarification
    assert plan_run.state == PlanRunState.NEED_CLARIFICATION
    outstanding_clarifications = plan_run.get_outstanding_clarifications()
    assert len(outstanding_clarifications) == 1

    clarification = outstanding_clarifications[0]
    assert isinstance(clarification, ActionClarification)
    assert clarification.user_guidance == "Please authenticate with the tool"
    assert str(clarification.action_url) == "https://example.com/auth"
    assert clarification.resolved is False

    ready_tool.mark_ready()
    portia.resolve_clarification(clarification, "Authentication completed", plan_run)

    result = await portia.aresume(plan_run, plan=plan)

    assert result.state == PlanRunState.COMPLETE
    assert result.outputs.final_output is not None
    assert result.outputs.final_output.get_value() == "Tool executed successfully"


@pytest.mark.asyncio
async def test_portia_plan_v2_tool_error_in_invoke_tool_step() -> None:
    """Test PlanV2 run_plan flow when a tool raises an error during execution."""
    error_tool = ErrorTool()

    config = Config.from_default(
        openai_api_key=SecretStr("123"),
    )
    tool_registry = ToolRegistry([error_tool])
    portia = Portia(config=config, tools=tool_registry)

    plan = (
        PlanBuilderV2("Test tool error in invoke_tool_step")
        .invoke_tool_step(
            step_name="Use error tool",
            tool="error_tool",
            args={},
        )
        .build()
    )

    # Run the plan - should fail due to tool error
    end_user = await portia.ainitialize_end_user()
    plan_run = await portia.arun_plan(plan, end_user)

    # Verify plan failed due to tool error
    assert plan_run.state == PlanRunState.FAILED
