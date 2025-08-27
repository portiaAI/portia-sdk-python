"""Test the Portia class with PlanV2."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field, HttpUrl, SecretStr

from portia.builder.plan_builder_v2 import PlanBuilderV2
from portia.builder.reference import Input
from portia.clarification import ActionClarification, ClarificationCategory
from portia.clarification_handler import ClarificationHandler
from portia.config import Config, ExecutionAgentType, GenerativeModelsConfig, StorageClass
from portia.errors import InvalidPlanRunStateError, ToolNotFoundError
from portia.execution_agents.default_execution_agent import DefaultExecutionAgent
from portia.execution_agents.one_shot_agent import OneShotAgent
from portia.execution_hooks import BeforeStepExecutionOutcome, ExecutionHooks
from portia.plan import Plan, PlanContext, PlanInput, Step
from portia.plan_run import PlanRun, PlanRunState
from portia.portia import Portia
from portia.tool import ReadyResponse, Tool, ToolRunContext, _ArgsSchemaPlaceholder
from portia.tool_registry import ToolRegistry
from portia.tool_wrapper import ToolCallWrapper
from tests.utils import AdditionTool, ClarificationTool, get_test_plan_run

if TYPE_CHECKING:
    from collections.abc import Callable

    from portia.builder.plan_v2 import PlanV2
    from portia.clarification import Clarification


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


def _create_portia_with_storage(
    storage_type: str, planning_model: MagicMock, tmp_dir: str | None = None
) -> Portia:
    """Create a Portia instance with the specified storage type."""
    if storage_type == "disk":
        if tmp_dir is None:
            msg = "tmp_dir is required for disk storage"
            raise ValueError(msg)
        config = Config.from_default(
            storage_class=StorageClass.DISK,
            openai_api_key=SecretStr("123"),
            storage_dir=tmp_dir,
            models=GenerativeModelsConfig(
                planning_model=planning_model,
            ),
        )
    else:
        config = Config.from_default(
            openai_api_key=SecretStr("123"),
            models=GenerativeModelsConfig(
                planning_model=planning_model,
            ),
        )
    tool_registry = ToolRegistry([AdditionTool(), ClarificationTool()])
    return Portia(config=config, tools=tool_registry)


@pytest.mark.parametrize("storage_type", ["local", "disk"])
def test_portia_run_plan_v2_sync_mainline(storage_type: str, planning_model: MagicMock) -> None:
    """Test running a PlanV2 synchronously works as expected for mainline case."""
    plan = (
        PlanBuilderV2("Test sync mainline")
        .function_step(function=lambda: "Step 1 result")
        .function_step(function=lambda: "Step 2 result")
        .build()
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        portia = _create_portia_with_storage(
            storage_type, planning_model, tmp_dir if storage_type == "disk" else None
        )
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
async def test_portia_run_plan_v2_async_mainline(
    storage_type: str, planning_model: MagicMock
) -> None:
    """Test running a PlanV2 asynchronously works as expected for mainline case."""
    plan = (
        PlanBuilderV2("Test async mainline")
        .function_step(function=lambda: "Step 1 result")
        .function_step(function=lambda: "Step 2 result")
        .build()
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        portia = _create_portia_with_storage(
            storage_type, planning_model, tmp_dir if storage_type == "disk" else None
        )
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


class FinalOutputSchema(BaseModel):
    """Schema for final output testing."""

    result: str = Field(description="The result value")
    count: int = Field(description="The count value")


class OverriddenOutputSchema(BaseModel):
    """Schema for overridden final output testing."""

    message: str = Field(description="The message value")
    success: bool = Field(description="Success indicator")


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

    end_user = await portia.ainitialize_end_user()
    plan_run = await portia.arun_plan(plan, end_user, structured_output_schema=runtime_schema)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.plan_id == plan.id
    assert plan_run.structured_output_schema == expected_schema_type
    assert plan_run.outputs.final_output is not None
    assert plan_run.outputs.final_output.get_value() == "test result"


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
    portia = Portia(config=Config.from_default(), tools=tool_registry)

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

    # Resume should work even from NOT_STARTED state
    result = await portia.aresume(plan_run, plan=plan)

    assert result.state == PlanRunState.COMPLETE
    assert result.outputs.final_output is not None
    assert result.outputs.final_output.get_value() == 15  # 10 + 5


@pytest.mark.parametrize("use_clarification_handler", [True, False])
def test_portia_resume_plan_v2_after_clarification_success(
    use_clarification_handler: bool,
) -> None:
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
    tool_registry = ToolRegistry([ClarificationTool()])
    portia = Portia(config=Config.from_default(), tools=tool_registry)
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


@pytest.mark.asyncio
async def test_portia_resume_plan_v2_invalid_state(portia: Portia) -> None:
    """Test running a PlanV2 asynchronously from an invalid state."""
    plan = (
        PlanBuilderV2("Test invalid state async")
        .function_step(function=lambda: "Step result")
        .build()
    )
    end_user = portia.initialize_end_user()

    legacy_plan = plan.to_legacy_plan(
        PlanContext(
            query=plan.label,
            tool_ids=[tool.id for tool in portia.tool_registry.get_tools()],
        )
    )
    plan_run = await portia._aget_plan_run_from_plan(legacy_plan, end_user, None)

    plan_run.state = PlanRunState.FAILED
    result = await portia.aresume(plan_run, plan=plan)

    assert result is plan_run
    assert result.state == PlanRunState.FAILED


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


def test_portia_get_tool_with_valid_tool_id(portia: Portia) -> None:
    """Test get_tool with a valid tool_id returns wrapped tool."""
    plan, plan_run = get_test_plan_run()

    tool = portia.get_tool("add_tool", plan_run)

    assert tool is not None
    assert isinstance(tool, ToolCallWrapper)
    assert isinstance(tool._child_tool, AdditionTool)  # pyright: ignore[reportAttributeAccessIssue]
    assert tool._storage == portia.storage  # pyright: ignore[reportAttributeAccessIssue]
    assert tool._plan_run == plan_run  # pyright: ignore[reportAttributeAccessIssue]


def test_portia_get_tool_with_none_tool_id(portia: Portia) -> None:
    """Test get_tool with None tool_id returns None."""
    plan, plan_run = get_test_plan_run()

    tool = portia.get_tool(None, plan_run)

    assert tool is None


def test_portia_get_tool_with_nonexistent_tool_id(portia: Portia) -> None:
    """Test get_tool with nonexistent tool_id raises ToolNotFoundError."""
    plan, plan_run = get_test_plan_run()

    with pytest.raises(ToolNotFoundError):
        portia.get_tool("nonexistent_tool", plan_run)


def test_portia_get_agent_for_step_with_default_execution_agent(portia: Portia) -> None:
    """Test get_agent_for_step returns DefaultExecutionAgent with DEFAULT config."""
    plan, plan_run = get_test_plan_run()
    step = Step(
        task="Add two numbers",
        inputs=[],
        output="$output",
        tool_id="add_tool",
    )

    # Ensure config is set to DEFAULT
    portia.config.execution_agent_type = ExecutionAgentType.DEFAULT

    agent = portia.get_agent_for_step(step, plan, plan_run)

    assert isinstance(agent, DefaultExecutionAgent)
    assert agent.plan == plan
    assert agent.plan_run == plan_run
    assert agent.config == portia.config


def test_portia_get_agent_for_step_with_oneshot_execution_agent(portia: Portia) -> None:
    """Test get_agent_for_step returns OneShotAgent with ONE_SHOT config."""
    plan, plan_run = get_test_plan_run()
    step = Step(
        task="Add two numbers",
        inputs=[],
        output="$output",
        tool_id="add_tool",
    )

    # Set config to ONE_SHOT
    portia.config.execution_agent_type = ExecutionAgentType.ONE_SHOT

    agent = portia.get_agent_for_step(step, plan, plan_run)

    assert isinstance(agent, OneShotAgent)
    assert agent.plan == plan
    assert agent.plan_run == plan_run
    assert agent.config == portia.config


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

    # Convert to legacy plan and create plan run
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

    # Convert to legacy plan and create plan run
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
