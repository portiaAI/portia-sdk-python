"""Test tool call serialization."""

from pydantic import BaseModel

from portia.prefixed_uuid import PlanRunUUID
from portia.tool_call import ToolCallRecord, ToolCallStatus


class DummyInput(BaseModel):
    """Input for testing."""

    value: str


class UnserializableOutput:
    """Input for testing."""

    def __init__(self, data) -> None:  # noqa: ANN001, D107
        self.data = data


def test_serializes_basemodel_input_and_output() -> None:
    """Test base models use pydantic."""
    input_obj = DummyInput(value="test")
    output_obj = DummyInput(value="result")

    record = ToolCallRecord(
        tool_name="test_tool",
        plan_run_id=PlanRunUUID(),
        step=1,
        end_user_id="user-abc",
        status=ToolCallStatus.SUCCESS,
        input=input_obj,
        output=output_obj,
        latency_seconds=0.1,
    )

    ser_input = record.serialize_input()
    ser_output = record.serialize_output()

    assert ser_input == {"value": "test"}
    assert ser_output == {"value": "result"}


def test_serializes_builtin_types_directly() -> None:
    """Built ins should be passed through."""
    record = ToolCallRecord(
        tool_name="echo",
        plan_run_id=PlanRunUUID(),
        step=2,
        end_user_id=None,
        status=ToolCallStatus.NEED_CLARIFICATION,
        input={"message": "hello"},
        output=["a", "b", "c"],
        latency_seconds=0.2,
    )

    ser_input = record.serialize_input()
    ser_output = record.serialize_output()

    assert ser_input == {"message": "hello"}
    assert ser_output == ["a", "b", "c"]


def test_unserializable_objects_are_flagged() -> None:
    """Check bad object."""
    record = ToolCallRecord(
        tool_name="fail_tool",
        plan_run_id=PlanRunUUID(),
        step=3,
        end_user_id=None,
        status=ToolCallStatus.FAILED,
        input="valid input",
        output=UnserializableOutput("oops"),
        latency_seconds=0.5,
    )

    ser_input = record.serialize_input()
    ser_output = record.serialize_output()

    assert ser_input == "valid input"
    assert ser_output.startswith("<<UNSERIALIZABLE: UnserializableOutput>>")
