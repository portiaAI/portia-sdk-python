"""Tests for RunContext."""

import pytest
from pydantic import ValidationError

from portia.builder.llm_step import LLMStep
from portia.builder.plan_v2 import PlanV2
from portia.config import Config, LLMProvider
from portia.end_user import EndUser
from portia.execution_agents.output import LocalDataValue, Output
from portia.execution_hooks import ExecutionHooks
from portia.plan_run import PlanRunState, PlanRunV2
from portia.run_context import RunContext, StepOutputValue
from portia.storage import InMemoryStorage
from portia.telemetry.telemetry_service import ProductTelemetry
from portia.tool_registry import InMemoryToolRegistry


@pytest.fixture
def sample_plan_v2() -> PlanV2:
    """Create a sample PlanV2 for testing."""
    return PlanV2(
        steps=[
            LLMStep(
                step_name="step1",
                prompt="Test prompt",
            ),
        ],
        label="Test Plan",
    )


@pytest.fixture
def sample_end_user() -> EndUser:
    """Create a sample EndUser for testing."""
    return EndUser(
        external_id="test_user_123",
        name="Test User",
        email="test@example.com",
    )


@pytest.fixture
def sample_config() -> Config:
    """Create a sample Config for testing."""
    return Config(
        llm_provider=LLMProvider.ANTHROPIC,
        anthropic_api_key="test_key",
    )


@pytest.fixture
def sample_plan_run_v2(
    sample_plan_v2: PlanV2, sample_end_user: EndUser, sample_config: Config
) -> PlanRunV2:
    """Create a sample PlanRunV2 for testing."""
    return PlanRunV2(
        plan=sample_plan_v2,
        end_user=sample_end_user,
        config=sample_config,
    )


@pytest.fixture
def sample_storage() -> InMemoryStorage:
    """Create a sample Storage for testing."""
    return InMemoryStorage()


@pytest.fixture
def sample_tool_registry() -> InMemoryToolRegistry:
    """Create a sample ToolRegistry for testing."""
    return InMemoryToolRegistry()


@pytest.fixture
def sample_execution_hooks() -> ExecutionHooks:
    """Create a sample ExecutionHooks for testing."""
    return ExecutionHooks()


@pytest.fixture
def sample_telemetry() -> ProductTelemetry:
    """Create a sample Telemetry for testing."""
    return ProductTelemetry()


@pytest.fixture
def run_context(
    sample_plan_run_v2: PlanRunV2,
    sample_storage: InMemoryStorage,
    sample_tool_registry: InMemoryToolRegistry,
    sample_execution_hooks: ExecutionHooks,
    sample_telemetry: ProductTelemetry,
) -> RunContext:
    """Create a RunContext instance for testing."""
    return RunContext(
        plan_run_v2=sample_plan_run_v2,
        storage=sample_storage,
        tool_registry=sample_tool_registry,
        execution_hooks=sample_execution_hooks,
        telemetry=sample_telemetry,
    )


def test_run_context_initialization(
    sample_plan_run_v2: PlanRunV2,
    sample_storage: InMemoryStorage,
    sample_tool_registry: InMemoryToolRegistry,
    sample_execution_hooks: ExecutionHooks,
    sample_telemetry: ProductTelemetry,
) -> None:
    """Test initialization of RunContext."""
    context = RunContext(
        plan_run_v2=sample_plan_run_v2,
        storage=sample_storage,
        tool_registry=sample_tool_registry,
        execution_hooks=sample_execution_hooks,
        telemetry=sample_telemetry,
    )

    assert context.plan_run_v2 == sample_plan_run_v2
    assert context.storage == sample_storage
    assert context.tool_registry == sample_tool_registry
    assert context.execution_hooks == sample_execution_hooks
    assert context.telemetry == sample_telemetry
    assert context.legacy_plan is None
    assert context.legacy_plan_run is None
    assert context.step_output_values == []


def test_run_context_required_fields(
    sample_plan_run_v2: PlanRunV2,
    sample_storage: InMemoryStorage,
    sample_tool_registry: InMemoryToolRegistry,
    sample_execution_hooks: ExecutionHooks,
    sample_telemetry: ProductTelemetry,
) -> None:
    """Test that required fields must be provided."""
    # Missing plan_run_v2 should raise ValidationError
    with pytest.raises(ValidationError):
        RunContext(
            storage=sample_storage,
            tool_registry=sample_tool_registry,
            execution_hooks=sample_execution_hooks,
            telemetry=sample_telemetry,
        )

    # Missing storage should raise ValidationError
    with pytest.raises(ValidationError):
        RunContext(
            plan_run_v2=sample_plan_run_v2,
            tool_registry=sample_tool_registry,
            execution_hooks=sample_execution_hooks,
            telemetry=sample_telemetry,
        )

    # Missing tool_registry should raise ValidationError
    with pytest.raises(ValidationError):
        RunContext(
            plan_run_v2=sample_plan_run_v2,
            storage=sample_storage,
            execution_hooks=sample_execution_hooks,
            telemetry=sample_telemetry,
        )

    # Missing execution_hooks should raise ValidationError
    with pytest.raises(ValidationError):
        RunContext(
            plan_run_v2=sample_plan_run_v2,
            storage=sample_storage,
            tool_registry=sample_tool_registry,
            telemetry=sample_telemetry,
        )

    # Missing telemetry should raise ValidationError
    with pytest.raises(ValidationError):
        RunContext(
            plan_run_v2=sample_plan_run_v2,
            storage=sample_storage,
            tool_registry=sample_tool_registry,
            execution_hooks=sample_execution_hooks,
        )


def test_run_context_with_legacy_fields(
    sample_plan_run_v2: PlanRunV2,
    sample_storage: InMemoryStorage,
    sample_tool_registry: InMemoryToolRegistry,
    sample_execution_hooks: ExecutionHooks,
    sample_telemetry: ProductTelemetry,
) -> None:
    """Test RunContext with legacy fields for backward compatibility."""
    from portia.plan import Plan, PlanContext, PlanRun

    legacy_plan_context = PlanContext(query="Test query", tool_registry=sample_tool_registry)
    legacy_plan = sample_plan_run_v2.plan.to_legacy_plan(legacy_plan_context)
    legacy_plan_run = PlanRun(
        plan_id=sample_plan_run_v2.plan.id,
        end_user_id=sample_plan_run_v2.end_user.external_id,
    )

    context = RunContext(
        plan_run_v2=sample_plan_run_v2,
        storage=sample_storage,
        tool_registry=sample_tool_registry,
        execution_hooks=sample_execution_hooks,
        telemetry=sample_telemetry,
        legacy_plan=legacy_plan,
        legacy_plan_run=legacy_plan_run,
    )

    assert context.legacy_plan == legacy_plan
    assert context.legacy_plan_run == legacy_plan_run


def test_run_context_get_tool_run_ctx_with_legacy_fields(
    sample_plan_run_v2: PlanRunV2,
    sample_storage: InMemoryStorage,
    sample_tool_registry: InMemoryToolRegistry,
    sample_execution_hooks: ExecutionHooks,
    sample_telemetry: ProductTelemetry,
) -> None:
    """Test get_tool_run_ctx with legacy fields."""
    from portia.plan import Plan, PlanContext, PlanRun

    legacy_plan_context = PlanContext(query="Test query", tool_registry=sample_tool_registry)
    legacy_plan = sample_plan_run_v2.plan.to_legacy_plan(legacy_plan_context)
    legacy_plan_run = PlanRun(
        plan_id=sample_plan_run_v2.plan.id,
        end_user_id=sample_plan_run_v2.end_user.external_id,
    )

    context = RunContext(
        plan_run_v2=sample_plan_run_v2,
        storage=sample_storage,
        tool_registry=sample_tool_registry,
        execution_hooks=sample_execution_hooks,
        telemetry=sample_telemetry,
        legacy_plan=legacy_plan,
        legacy_plan_run=legacy_plan_run,
    )

    tool_run_ctx = context.get_tool_run_ctx()
    assert tool_run_ctx.end_user == sample_plan_run_v2.end_user
    assert tool_run_ctx.plan_run == legacy_plan_run
    assert tool_run_ctx.plan == legacy_plan
    assert tool_run_ctx.config == sample_plan_run_v2.config


def test_run_context_get_tool_run_ctx_without_legacy_fields(run_context: RunContext) -> None:
    """Test get_tool_run_ctx without legacy fields (fallback mode)."""
    tool_run_ctx = run_context.get_tool_run_ctx()
    assert tool_run_ctx.end_user == run_context.plan_run_v2.end_user
    assert tool_run_ctx.config == run_context.plan_run_v2.config
    assert tool_run_ctx.plan_run is not None
    assert tool_run_ctx.plan is not None


def test_run_context_step_output_values(run_context: RunContext) -> None:
    """Test step_output_values field."""
    assert run_context.step_output_values == []

    # Add a step output value
    step_output = StepOutputValue(
        value="test_value",
        description="Test description",
        step_name="step1",
        step_num=0,
    )
    run_context.step_output_values.append(step_output)

    assert len(run_context.step_output_values) == 1
    assert run_context.step_output_values[0] == step_output


def test_run_context_plan_run_v2_access(run_context: RunContext) -> None:
    """Test accessing PlanRunV2 fields through RunContext."""
    assert run_context.plan_run_v2.state == PlanRunState.NOT_STARTED
    assert run_context.plan_run_v2.current_step_index == 0
    assert run_context.plan_run_v2.plan is not None
    assert run_context.plan_run_v2.end_user is not None
    assert run_context.plan_run_v2.config is not None


def test_run_context_plan_run_v2_mutation(run_context: RunContext) -> None:
    """Test mutating PlanRunV2 through RunContext."""
    run_context.plan_run_v2.state = PlanRunState.IN_PROGRESS
    assert run_context.plan_run_v2.state == PlanRunState.IN_PROGRESS

    run_context.plan_run_v2.current_step_index = 5
    assert run_context.plan_run_v2.current_step_index == 5


def test_run_context_serialization(run_context: RunContext) -> None:
    """Test RunContext can be serialized to JSON."""
    # Add some data
    run_context.plan_run_v2.state = PlanRunState.IN_PROGRESS
    run_context.plan_run_v2.current_step_index = 2
    run_context.step_output_values.append(
        StepOutputValue(value="test", description="desc", step_name="step1", step_num=0)
    )

    # Serialize to dict (JSON serialization requires special handling for some types)
    data = run_context.model_dump()
    assert data is not None
    assert "plan_run_v2" in data
    assert "storage" in data
    assert "tool_registry" in data
    assert "execution_hooks" in data
    assert "telemetry" in data


def test_step_output_value_creation() -> None:
    """Test StepOutputValue creation."""
    output_value = StepOutputValue(
        value="test_value",
        description="Test description",
        step_name="test_step",
        step_num=1,
    )

    assert output_value.value == "test_value"
    assert output_value.description == "Test description"
    assert output_value.step_name == "test_step"
    assert output_value.step_num == 1


def test_step_output_value_default_description() -> None:
    """Test StepOutputValue with default description."""
    output_value = StepOutputValue(
        value="test_value",
        step_name="test_step",
        step_num=1,
    )

    assert output_value.description == ""


def test_step_output_value_with_complex_value() -> None:
    """Test StepOutputValue with complex value types."""
    # Test with dict
    dict_output = StepOutputValue(
        value={"key": "value", "nested": {"inner": 123}},
        step_name="dict_step",
        step_num=0,
    )
    assert dict_output.value["key"] == "value"
    assert dict_output.value["nested"]["inner"] == 123

    # Test with list
    list_output = StepOutputValue(
        value=[1, 2, 3, "four"],
        step_name="list_step",
        step_num=1,
    )
    assert list_output.value == [1, 2, 3, "four"]

    # Test with None
    none_output = StepOutputValue(
        value=None,
        step_name="none_step",
        step_num=2,
    )
    assert none_output.value is None