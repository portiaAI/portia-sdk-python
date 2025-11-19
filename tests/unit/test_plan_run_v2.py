"""Tests for PlanRunV2 primitives."""

import pytest
from pydantic import ValidationError

from portia.builder.plan_v2 import PlanV2
from portia.builder.step_v2 import StepV2
from portia.builder.llm_step import LLMStep
from portia.config import Config, LLMProvider
from portia.end_user import EndUser
from portia.execution_agents.output import LocalDataValue, Output
from portia.plan_run import PlanRunState, PlanRunV2
from portia.prefixed_uuid import PlanRunUUID


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
def plan_run_v2(sample_plan_v2: PlanV2, sample_end_user: EndUser, sample_config: Config) -> PlanRunV2:
    """Create a PlanRunV2 instance for testing."""
    return PlanRunV2(
        plan=sample_plan_v2,
        end_user=sample_end_user,
        config=sample_config,
    )


def test_plan_run_v2_initialization(
    sample_plan_v2: PlanV2, sample_end_user: EndUser, sample_config: Config
) -> None:
    """Test initialization of PlanRunV2 instance."""
    plan_run_inputs = {"$input1": LocalDataValue(value="test_input_value")}
    plan_run = PlanRunV2(
        plan=sample_plan_v2,
        end_user=sample_end_user,
        config=sample_config,
        plan_run_inputs=plan_run_inputs,
    )

    assert plan_run.id is not None
    assert isinstance(plan_run.id, PlanRunUUID)
    assert plan_run.plan == sample_plan_v2
    assert plan_run.end_user == sample_end_user
    assert plan_run.config == sample_config
    assert plan_run.current_step_index == 0
    assert plan_run.state == PlanRunState.NOT_STARTED
    assert plan_run.step_output_values == []
    assert plan_run.final_output is None
    assert len(plan_run.plan_run_inputs) == 1
    assert plan_run.plan_run_inputs["$input1"].get_value() == "test_input_value"


def test_plan_run_v2_with_custom_id(
    sample_plan_v2: PlanV2, sample_end_user: EndUser, sample_config: Config
) -> None:
    """Test PlanRunV2 can be created with custom ID."""
    custom_id = PlanRunUUID()
    plan_run = PlanRunV2(
        id=custom_id,
        plan=sample_plan_v2,
        end_user=sample_end_user,
        config=sample_config,
    )

    assert plan_run.id == custom_id


def test_plan_run_v2_state_transitions(plan_run_v2: PlanRunV2) -> None:
    """Test state transitions of PlanRunV2."""
    assert plan_run_v2.state == PlanRunState.NOT_STARTED

    plan_run_v2.state = PlanRunState.IN_PROGRESS
    assert plan_run_v2.state == PlanRunState.IN_PROGRESS

    plan_run_v2.state = PlanRunState.COMPLETE
    assert plan_run_v2.state == PlanRunState.COMPLETE


def test_plan_run_v2_current_step_index(plan_run_v2: PlanRunV2) -> None:
    """Test current_step_index tracking."""
    assert plan_run_v2.current_step_index == 0

    plan_run_v2.current_step_index = 1
    assert plan_run_v2.current_step_index == 1

    plan_run_v2.current_step_index = 5
    assert plan_run_v2.current_step_index == 5


def test_plan_run_v2_step_output_values(plan_run_v2: PlanRunV2) -> None:
    """Test step output values storage."""
    assert plan_run_v2.step_output_values == []

    output1 = {"result": Output(value="output1")}
    plan_run_v2.step_output_values.append(output1)
    assert len(plan_run_v2.step_output_values) == 1
    assert plan_run_v2.step_output_values[0] == output1

    output2 = {"result": Output(value="output2")}
    plan_run_v2.step_output_values.append(output2)
    assert len(plan_run_v2.step_output_values) == 2
    assert plan_run_v2.step_output_values[1] == output2


def test_plan_run_v2_final_output(plan_run_v2: PlanRunV2) -> None:
    """Test final output storage."""
    assert plan_run_v2.final_output is None

    final_output = Output(value="Final result")
    plan_run_v2.final_output = final_output
    assert plan_run_v2.final_output == final_output


def test_plan_run_v2_plan_run_inputs(
    sample_plan_v2: PlanV2, sample_end_user: EndUser, sample_config: Config
) -> None:
    """Test plan_run_inputs storage and retrieval."""
    inputs = {
        "$input1": LocalDataValue(value="value1"),
        "$input2": LocalDataValue(value=42),
        "$input3": LocalDataValue(value={"key": "value"}),
    }
    plan_run = PlanRunV2(
        plan=sample_plan_v2,
        end_user=sample_end_user,
        config=sample_config,
        plan_run_inputs=inputs,
    )

    assert len(plan_run.plan_run_inputs) == 3
    assert plan_run.plan_run_inputs["$input1"].get_value() == "value1"
    assert plan_run.plan_run_inputs["$input2"].get_value() == 42
    assert plan_run.plan_run_inputs["$input3"].get_value() == {"key": "value"}


def test_plan_run_v2_str_representation(plan_run_v2: PlanRunV2) -> None:
    """Test string representation of PlanRunV2."""
    str_repr = str(plan_run_v2)
    assert "PlanRunV2" in str_repr
    assert str(plan_run_v2.id) in str_repr
    assert str(plan_run_v2.plan.id) in str_repr
    assert str(plan_run_v2.state) in str_repr
    assert str(plan_run_v2.current_step_index) in str_repr
    assert "unset" in str_repr  # final_output is None

    # Test with final_output set
    plan_run_v2.final_output = Output(value="result")
    str_repr = str(plan_run_v2)
    assert "set" in str_repr


def test_plan_run_v2_serialization(plan_run_v2: PlanRunV2) -> None:
    """Test PlanRunV2 can be serialized to JSON and deserialized."""
    # Add some data to test serialization
    plan_run_v2.state = PlanRunState.IN_PROGRESS
    plan_run_v2.current_step_index = 2
    plan_run_v2.step_output_values = [{"result": Output(value="test")}]
    plan_run_v2.final_output = Output(value="final")
    plan_run_v2.plan_run_inputs = {"$input": LocalDataValue(value="test_value")}

    # Serialize to JSON
    json_str = plan_run_v2.model_dump_json()
    assert json_str is not None
    assert len(json_str) > 0

    # Deserialize from JSON
    parsed_plan_run = PlanRunV2.model_validate_json(json_str)
    assert parsed_plan_run.id == plan_run_v2.id
    assert parsed_plan_run.state == plan_run_v2.state
    assert parsed_plan_run.current_step_index == plan_run_v2.current_step_index
    assert parsed_plan_run.plan.id == plan_run_v2.plan.id
    assert parsed_plan_run.end_user.external_id == plan_run_v2.end_user.external_id
    assert len(parsed_plan_run.step_output_values) == 1
    assert parsed_plan_run.final_output is not None
    assert parsed_plan_run.plan_run_inputs["$input"].get_value() == "test_value"


def test_plan_run_v2_required_fields(sample_end_user: EndUser, sample_config: Config) -> None:
    """Test that required fields must be provided."""
    # Missing plan should raise ValidationError
    with pytest.raises(ValidationError):
        PlanRunV2(
            end_user=sample_end_user,
            config=sample_config,
        )

    # Missing end_user should raise ValidationError
    with pytest.raises(ValidationError):
        PlanRunV2(
            plan=PlanV2(steps=[], label="Test"),
            config=sample_config,
        )

    # Missing config should raise ValidationError
    with pytest.raises(ValidationError):
        PlanRunV2(
            plan=PlanV2(steps=[], label="Test"),
            end_user=sample_end_user,
        )


def test_plan_run_v2_immutable_after_creation(plan_run_v2: PlanRunV2) -> None:
    """Test that PlanRunV2 fields can be mutated (it's not frozen)."""
    # PlanRunV2 is not frozen, so we should be able to mutate it
    original_state = plan_run_v2.state
    plan_run_v2.state = PlanRunState.IN_PROGRESS
    assert plan_run_v2.state == PlanRunState.IN_PROGRESS
    assert plan_run_v2.state != original_state


def test_plan_run_v2_with_empty_step_outputs(
    sample_plan_v2: PlanV2, sample_end_user: EndUser, sample_config: Config
) -> None:
    """Test PlanRunV2 with empty step outputs."""
    plan_run = PlanRunV2(
        plan=sample_plan_v2,
        end_user=sample_end_user,
        config=sample_config,
        step_output_values=[],
    )

    assert plan_run.step_output_values == []


def test_plan_run_v2_no_extra_fields_allowed(
    sample_plan_v2: PlanV2, sample_end_user: EndUser, sample_config: Config
) -> None:
    """Test that extra fields are not allowed in PlanRunV2."""
    with pytest.raises(ValidationError):
        PlanRunV2(
            plan=sample_plan_v2,
            end_user=sample_end_user,
            config=sample_config,
            extra_field="not_allowed",  # type: ignore
        )


def test_plan_run_v2_end_user_integration(
    sample_plan_v2: PlanV2, sample_config: Config
) -> None:
    """Test PlanRunV2 with different EndUser configurations."""
    # Test with minimal EndUser
    minimal_user = EndUser(external_id="minimal_user")
    plan_run = PlanRunV2(
        plan=sample_plan_v2,
        end_user=minimal_user,
        config=sample_config,
    )
    assert plan_run.end_user.external_id == "minimal_user"
    assert plan_run.end_user.name == ""
    assert plan_run.end_user.email == ""

    # Test with full EndUser
    full_user = EndUser(
        external_id="full_user",
        name="Full User",
        email="full@example.com",
        phone_number="+1234567890",
        additional_data={"key": "value"},
    )
    plan_run = PlanRunV2(
        plan=sample_plan_v2,
        end_user=full_user,
        config=sample_config,
    )
    assert plan_run.end_user.external_id == "full_user"
    assert plan_run.end_user.name == "Full User"
    assert plan_run.end_user.email == "full@example.com"
    assert plan_run.end_user.phone_number == "+1234567890"
    assert plan_run.end_user.additional_data == {"key": "value"}