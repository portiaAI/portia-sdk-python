"""Tests for RunContext primitives and migrations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from portia.builder.plan_v2 import PlanV2
from portia.builder.step_v2 import StepV2
from portia.clarification import InputClarification
from portia.config import Config
from portia.end_user import EndUser
from portia.execution_agents.output import LocalDataValue
from portia.execution_hooks import ExecutionHooks
from portia.plan import Plan, PlanContext
from portia.plan_run import PlanRun, PlanRunOutputs, PlanRunState, PlanRunV2
from portia.prefixed_uuid import PlanRunUUID, PlanUUID
from portia.run_context import (
    RunContext,
    RunContextV2,
    StepOutputValue,
    migrate_run_context_to_v2,
    migrate_v2_to_run_context,
)
from portia.storage import Storage
from portia.telemetry.telemetry_service import BaseProductTelemetry
from portia.tool_registry import ToolRegistry


@pytest.fixture
def mock_end_user() -> EndUser:
    """Create a mock EndUser for testing."""
    return EndUser(external_id="test_user_123", name="Test User", email="test@example.com")


@pytest.fixture
def mock_config() -> Config:
    """Create a mock Config for testing."""
    return Config(
        llm_provider=None,
        storage_class="MEMORY",
    )


@pytest.fixture
def mock_plan_v2() -> PlanV2:
    """Create a mock PlanV2 for testing."""
    return PlanV2(
        id=PlanUUID(),
        steps=[
            StepV2(step_name="test_step", task="Test task"),
        ],
        label="Test Plan V2"
    )


@pytest.fixture
def mock_legacy_plan() -> Plan:
    """Create a mock legacy Plan for testing."""
    from portia.plan import Step

    return Plan(
        id=PlanUUID(),
        plan_context=PlanContext(query="Test query"),
        steps=[
            Step(task="Test task", output="$output"),
        ],
    )


@pytest.fixture
def mock_storage() -> Storage:
    """Create a mock Storage for testing."""
    from portia.storage import InMemoryStorage
    return InMemoryStorage()


@pytest.fixture
def mock_tool_registry() -> ToolRegistry:
    """Create a mock ToolRegistry for testing."""
    return ToolRegistry()


@pytest.fixture
def mock_execution_hooks() -> ExecutionHooks:
    """Create a mock ExecutionHooks for testing."""
    return ExecutionHooks()


@pytest.fixture
def mock_telemetry() -> BaseProductTelemetry:
    """Create a mock BaseProductTelemetry for testing."""
    from portia.telemetry.telemetry_service import NoOpTelemetry
    return NoOpTelemetry()


@pytest.fixture
def mock_plan_run(mock_legacy_plan, mock_end_user) -> PlanRun:
    """Create a mock PlanRun for testing."""
    return PlanRun(
        plan_id=mock_legacy_plan.id,
        end_user_id=mock_end_user.external_id,
        current_step_index=1,
        state=PlanRunState.IN_PROGRESS,
        outputs=PlanRunOutputs(
            step_outputs={"step1": LocalDataValue(value="test_output")},
        ),
        plan_run_inputs={"$input1": LocalDataValue(value="test_input")},
    )


@pytest.fixture
def legacy_run_context(
    mock_plan_v2,
    mock_legacy_plan,
    mock_plan_run,
    mock_end_user,
    mock_config,
    mock_storage,
    mock_tool_registry,
    mock_execution_hooks,
    mock_telemetry,
) -> RunContext:
    """Create a legacy RunContext for testing."""
    return RunContext(
        plan=mock_plan_v2,
        legacy_plan=mock_legacy_plan,
        plan_run=mock_plan_run,
        end_user=mock_end_user,
        step_output_values=[StepOutputValue(value="test", description="test", step_name="test", step_num=1)],
        config=mock_config,
        storage=mock_storage,
        tool_registry=mock_tool_registry,
        execution_hooks=mock_execution_hooks,
        telemetry=mock_telemetry,
    )


@pytest.fixture
def plan_run_v2(mock_end_user, mock_config, mock_plan_v2) -> PlanRunV2:
    """Create PlanRunV2 instance for testing."""
    return PlanRunV2(
        end_user=mock_end_user,
        config=mock_config,
        plan=mock_plan_v2,
        current_step_index=1,
        state=PlanRunState.IN_PROGRESS,
        step_output_values=[LocalDataValue(value="test_output")],
        plan_run_inputs={"$input1": LocalDataValue(value="test_input")},
    )


@pytest.fixture
def run_context_v2(
    plan_run_v2,
    mock_storage,
    mock_tool_registry,
    mock_execution_hooks,
    mock_telemetry,
) -> RunContextV2:
    """Create RunContextV2 instance for testing."""
    return RunContextV2(
        plan_run=plan_run_v2,
        storage=mock_storage,
        tool_registry=mock_tool_registry,
        execution_hooks=mock_execution_hooks,
        telemetry=mock_telemetry,
    )


def test_step_output_value_creation():
    """Test creation of StepOutputValue."""
    step_output = StepOutputValue(
        value="test_value",
        description="test description",
        step_name="test_step",
        step_num=1
    )

    assert step_output.value == "test_value"
    assert step_output.description == "test description"
    assert step_output.step_name == "test_step"
    assert step_output.step_num == 1


def test_legacy_run_context_initialization(legacy_run_context: RunContext):
    """Test initialization of legacy RunContext."""
    assert legacy_run_context.plan is not None
    assert legacy_run_context.legacy_plan is not None
    assert legacy_run_context.plan_run is not None
    assert legacy_run_context.end_user is not None
    assert legacy_run_context.config is not None
    assert legacy_run_context.storage is not None
    assert legacy_run_context.tool_registry is not None
    assert legacy_run_context.execution_hooks is not None
    assert legacy_run_context.telemetry is not None
    assert len(legacy_run_context.step_output_values) == 1


def test_legacy_run_context_get_tool_run_ctx(legacy_run_context: RunContext):
    """Test get_tool_run_ctx method of legacy RunContext."""
    tool_run_ctx = legacy_run_context.get_tool_run_ctx()

    assert tool_run_ctx.end_user == legacy_run_context.end_user
    assert tool_run_ctx.plan_run == legacy_run_context.plan_run
    assert tool_run_ctx.plan == legacy_run_context.legacy_plan
    assert tool_run_ctx.config == legacy_run_context.config


def test_run_context_v2_initialization(run_context_v2: RunContextV2):
    """Test initialization of RunContextV2."""
    assert run_context_v2.plan_run is not None
    assert run_context_v2.storage is not None
    assert run_context_v2.tool_registry is not None
    assert run_context_v2.execution_hooks is not None
    assert run_context_v2.telemetry is not None


def test_run_context_v2_get_tool_run_ctx(run_context_v2: RunContextV2):
    """Test get_tool_run_ctx method of RunContextV2."""
    tool_run_ctx = run_context_v2.get_tool_run_ctx()

    assert tool_run_ctx.end_user == run_context_v2.plan_run.end_user
    assert tool_run_ctx.config == run_context_v2.plan_run.config
    # The plan_run should be a converted legacy plan run
    assert tool_run_ctx.plan_run.id == run_context_v2.plan_run.id
    assert tool_run_ctx.plan_run.current_step_index == run_context_v2.plan_run.current_step_index


def test_run_context_v2_get_tool_run_ctx_without_plan(
    mock_end_user,
    mock_config,
    mock_storage,
    mock_tool_registry,
    mock_execution_hooks,
    mock_telemetry,
):
    """Test get_tool_run_ctx method when PlanRunV2 has no plan."""
    plan_run_v2 = PlanRunV2(
        end_user=mock_end_user,
        config=mock_config,
        plan=None,
    )

    run_context_v2 = RunContextV2(
        plan_run=plan_run_v2,
        storage=mock_storage,
        tool_registry=mock_tool_registry,
        execution_hooks=mock_execution_hooks,
        telemetry=mock_telemetry,
    )

    tool_run_ctx = run_context_v2.get_tool_run_ctx()

    assert tool_run_ctx.end_user == mock_end_user
    assert tool_run_ctx.config == mock_config
    assert tool_run_ctx.plan is None
    assert tool_run_ctx.plan_run.plan_id == "unknown"


# Migration Tests
def test_migrate_run_context_to_v2(legacy_run_context: RunContext):
    """Test migration from legacy RunContext to RunContextV2."""
    context_v2 = migrate_run_context_to_v2(legacy_run_context)

    # Check that all fields are migrated correctly
    assert context_v2.storage == legacy_run_context.storage
    assert context_v2.tool_registry == legacy_run_context.tool_registry
    assert context_v2.execution_hooks == legacy_run_context.execution_hooks
    assert context_v2.telemetry == legacy_run_context.telemetry

    # Check that PlanRunV2 is created with correct data
    assert context_v2.plan_run.id == legacy_run_context.plan_run.id
    assert context_v2.plan_run.state == legacy_run_context.plan_run.state
    assert context_v2.plan_run.current_step_index == legacy_run_context.plan_run.current_step_index
    assert context_v2.plan_run.plan == legacy_run_context.plan
    assert context_v2.plan_run.end_user == legacy_run_context.end_user
    assert context_v2.plan_run.config == legacy_run_context.config


def test_migrate_v2_to_run_context(run_context_v2: RunContextV2):
    """Test migration from RunContextV2 back to legacy RunContext."""
    legacy_context = migrate_v2_to_run_context(run_context_v2)

    # Check that all fields are migrated correctly
    assert legacy_context.storage == run_context_v2.storage
    assert legacy_context.tool_registry == run_context_v2.tool_registry
    assert legacy_context.execution_hooks == run_context_v2.execution_hooks
    assert legacy_context.telemetry == run_context_v2.telemetry

    # Check that legacy structures are recreated correctly
    assert legacy_context.plan_run.id == run_context_v2.plan_run.id
    assert legacy_context.plan_run.current_step_index == run_context_v2.plan_run.current_step_index
    assert legacy_context.plan_run.state == run_context_v2.plan_run.state
    assert legacy_context.plan == run_context_v2.plan_run.plan
    assert legacy_context.end_user == run_context_v2.plan_run.end_user
    assert legacy_context.config == run_context_v2.plan_run.config
    assert legacy_context.step_output_values == run_context_v2.plan_run.step_output_values


def test_run_context_migration_roundtrip(legacy_run_context: RunContext):
    """Test that RunContext migration works correctly in both directions."""
    # Migrate to V2 and back
    context_v2 = migrate_run_context_to_v2(legacy_run_context)
    migrated_back = migrate_v2_to_run_context(context_v2)

    # Check that key fields are preserved after roundtrip
    assert migrated_back.plan == legacy_run_context.plan
    assert migrated_back.end_user == legacy_run_context.end_user
    assert migrated_back.config == legacy_run_context.config
    assert migrated_back.storage == legacy_run_context.storage
    assert migrated_back.tool_registry == legacy_run_context.tool_registry
    assert migrated_back.execution_hooks == legacy_run_context.execution_hooks
    assert migrated_back.telemetry == legacy_run_context.telemetry

    # Check plan run fields
    assert migrated_back.plan_run.id == legacy_run_context.plan_run.id
    assert migrated_back.plan_run.current_step_index == legacy_run_context.plan_run.current_step_index
    assert migrated_back.plan_run.state == legacy_run_context.plan_run.state
    assert migrated_back.plan_run.plan_run_inputs == legacy_run_context.plan_run.plan_run_inputs


def test_run_context_migration_with_clarifications(legacy_run_context: RunContext):
    """Test RunContext migration preserves clarifications."""
    # Add a clarification to the legacy run context
    clarification = InputClarification(
        plan_run_id=legacy_run_context.plan_run.id,
        step=1,
        argument_name="test_arg",
        user_guidance="test guidance",
        resolved=False,
        source="Test plan run",
    )
    legacy_run_context.plan_run.outputs.clarifications = [clarification]

    # Migrate to V2 and back
    context_v2 = migrate_run_context_to_v2(legacy_run_context)
    migrated_back = migrate_v2_to_run_context(context_v2)

    # Check that clarifications are preserved
    assert len(migrated_back.plan_run.outputs.clarifications) == 1
    assert migrated_back.plan_run.outputs.clarifications[0] == clarification

    # Check that V2 context can access clarifications
    outstanding = context_v2.plan_run.get_outstanding_clarifications()
    assert len(outstanding) == 1
    assert outstanding[0] == clarification