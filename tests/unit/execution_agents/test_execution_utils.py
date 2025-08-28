"""Test execution utilities."""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, MessagesState

from portia.clarification import InputClarification
from portia.errors import (
    InvalidAgentOutputError,
    InvalidPlanRunStateError,
    ToolFailedError,
    ToolRetryError,
    ToolSoftError,
)
from portia.execution_agents.context import StepInput
from portia.execution_agents.execution_utils import (
    AgentNode,
    get_arg_value_with_templating,
    process_output,
    template_in_required_inputs,
    tool_call_or_end,
)
from portia.execution_agents.output import LocalDataValue, Output
from portia.plan import Step
from portia.prefixed_uuid import PlanRunUUID
from tests.utils import AdditionTool


@pytest.fixture
def step() -> MagicMock:
    """Fixture to create a mock planning model."""
    step = MagicMock(spec=Step)
    step.structured_output_schema = None
    return step


def test_tool_call_or_end() -> None:
    """Test tool_call_or_end state transitions."""
    message_with_calls = AIMessage(content="test")
    state_with_calls: MessagesState = {"messages": [message_with_calls]}  # type: ignore  # noqa: PGH003

    message_without_calls = HumanMessage(content="test")
    state_without_calls: MessagesState = {"messages": [message_without_calls]}  # type: ignore  # noqa: PGH003

    assert tool_call_or_end(state_with_calls) == AgentNode.TOOLS
    assert tool_call_or_end(state_without_calls) == END


def test_process_output_with_clarifications(step: MagicMock) -> None:
    """Test process_output with clarifications."""
    clarifications = [
        InputClarification(
            argument_name="test",
            user_guidance="test",
            plan_run_id=PlanRunUUID(),
            source="Test process output with clarifications",
        ),
    ]
    message = HumanMessage(content="test")

    result = process_output(step, [message], clarifications=clarifications)  # type: ignore  # noqa: PGH003

    assert isinstance(result, Output)
    assert result.get_value() == clarifications


def test_process_output_with_tool_errors(step: MagicMock) -> None:
    """Test process_output with tool errors."""
    tool = AdditionTool()

    soft_error = ToolMessage(
        content="Error: ToolSoftError(test)",
        tool_call_id="1",
        name="test",
        status="error",
    )
    hard_error = ToolMessage(
        content="Error: ToolHardError(test)",
        tool_call_id="1",
        name="test",
        status="error",
    )

    with pytest.raises(ToolRetryError):
        process_output(step, [soft_error], tool)

    with pytest.raises(ToolFailedError):
        process_output(step, [hard_error], tool)


def test_process_output_with_invalid_message(step: MagicMock) -> None:
    """Test process_output with invalid message."""
    invalid_message = AIMessage(content="test")

    with pytest.raises(InvalidAgentOutputError):
        process_output(step, [invalid_message])


def test_process_output_with_output_artifacts(step: MagicMock) -> None:
    """Test process_output with outpu artifacts."""
    message = ToolMessage(tool_call_id="1", content="", artifact=LocalDataValue(value="test"))
    message2 = ToolMessage(tool_call_id="2", content="", artifact=LocalDataValue(value="bar"))

    result = process_output(step, [message, message2], clarifications=[])

    assert isinstance(result, Output)
    assert result.get_value() == ["test", "bar"]
    assert result.get_summary() == "test, bar"


def test_process_output_with_artifacts(step: MagicMock) -> None:
    """Test process_output with artifacts."""
    message = ToolMessage(tool_call_id="1", content="", artifact="test")

    result = process_output(step, [message], clarifications=[])

    assert isinstance(result, Output)
    assert result.get_value() == "test"


def test_process_output_with_content(step: MagicMock) -> None:
    """Test process_output with content."""
    message = ToolMessage(tool_call_id="1", content="test")

    result = process_output(step, [message], clarifications=[])

    assert isinstance(result, Output)
    assert result.get_value() == "test"


def test_process_output_with_clarification(step: MagicMock) -> None:
    """Test process_output with a clarification."""
    clarification = InputClarification(
        argument_name="test",
        user_guidance="test",
        plan_run_id=PlanRunUUID(),
        source="Test process output with clarification",
    )
    message = ToolMessage(tool_call_id="1", content=clarification.model_dump_json())

    result = process_output(step, [message], clarifications=[])

    assert isinstance(result, Output)
    assert result.get_value() == [clarification]


def test_process_output_summary_matches_serialized_value(step: MagicMock) -> None:
    """Test process_output summary matches serialized value."""
    dict_value = {"key1": "value1", "key2": "value2"}
    message = ToolMessage(
        tool_call_id="1", content="test", artifact=LocalDataValue(value=dict_value)
    )

    result = process_output(step, [message], clarifications=[])

    assert isinstance(result, Output)
    assert result.get_value() == dict_value
    assert result.get_summary() == result.serialize_value()


def test_process_output_summary_not_updated_if_provided(step: MagicMock) -> None:
    """Test process_output does not update summary if already provided."""
    dict_value = {"key1": "value1", "key2": "value2"}
    provided_summary = "This is a provided summary."
    message = ToolMessage(
        tool_call_id="1",
        content="test",
        artifact=LocalDataValue(value=dict_value, summary=provided_summary),
    )

    result = process_output(step, [message], clarifications=[])

    assert isinstance(result, Output)
    assert result.get_value() == dict_value
    assert result.get_summary() == provided_summary


def test_get_arg_value_with_templating_no_templating() -> None:
    """Test get_arg_value_with_templating with an arg that needs no templating."""
    result = get_arg_value_with_templating([], "simple string")
    assert result == "simple string"


def test_get_arg_value_with_templating_string_with_templating() -> None:
    """Test get_arg_value_with_templating with a string arg that needs 2 values templated in."""
    step_inputs = [
        StepInput(name="$name", value="John", description="User's name"),
        StepInput(name="$age", value="30", description="User's age"),
    ]
    arg = "Hello {{$name}}, you are {{$age}} years old"

    result = get_arg_value_with_templating(step_inputs, arg)
    assert result == "Hello John, you are 30 years old"


def test_get_arg_value_with_templating_list_with_templating() -> None:
    """Test get_arg_value_with_templating with a list of strings that needs a value templated in."""
    step_inputs = [
        StepInput(name="$name", value="John", description="User's name"),
    ]
    arg = ["Hello {{$name}}", "Goodbye {{$name}}"]

    result = get_arg_value_with_templating(step_inputs, arg)
    assert result == ["Hello John", "Goodbye John"]


def test_get_arg_value_with_templating_dict_with_templating() -> None:
    """Test get_arg_value_with_templating with a dict of strings that needs a value templated in."""
    step_inputs = [
        StepInput(name="$name", value="John", description="User's name"),
    ]
    arg = {"greeting": "Hello {{$name}}", "farewell": "Goodbye {{$name}}"}

    result = get_arg_value_with_templating(step_inputs, arg)
    assert result == {"greeting": "Hello John", "farewell": "Goodbye John"}


def test_template_in_required_inputs_multiple_args() -> None:
    """Test template_in_required_inputs with two different args that need templating."""
    step_inputs = [
        StepInput(name="$name", value="John", description="User's name"),
        StepInput(name="$age", value="30", description="User's age"),
    ]
    message = AIMessage(content="")
    message.tool_calls = [
        {
            "name": "test_tool",
            "type": "tool_call",
            "id": "call_123",
            "args": {"greeting": "Hello {{$name}}", "age_info": "You are {{$age}} years old"},
        }
    ]

    result = template_in_required_inputs(message, step_inputs)

    assert result.tool_calls[0]["args"]["greeting"] == "Hello John"  # pyright: ignore[reportAttributeAccessIssue]
    assert result.tool_calls[0]["args"]["age_info"] == "You are 30 years old"  # pyright: ignore[reportAttributeAccessIssue]


def test_template_in_required_inputs_missing_args() -> None:
    """Test template_in_required_inputs with error case of a tool_call with no args field."""
    step_inputs = [
        StepInput(name="$name", value="John", description="User's name"),
    ]
    message = AIMessage(content="")
    message.tool_calls = [{"name": "test_tool", "type": "tool_call", "id": "call_123"}]  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(InvalidPlanRunStateError, match="Tool call missing args field"):
        template_in_required_inputs(message, step_inputs)


def test_template_in_required_inputs_extra_var_raises_error() -> None:
    """Test template_in_required_inputs with an extra variable in the template."""
    step_inputs = [
        StepInput(name="$name", value="John", description="User's name"),
    ]
    message = AIMessage(content="")
    message.tool_calls = [
        {
            "name": "test_tool",
            "type": "tool_call",
            "id": "call_123",
            "args": {"greeting": "Hello {{$name}}, you are {{$age}} years old"},
        }
    ]  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(
        ToolSoftError,
        match=re.escape(
            "Templated variables found in input that are not valid inputs for step: $age"
        ),
    ):
        template_in_required_inputs(message, step_inputs)


def test_process_output_with_mixed_list_and_scalar_values(step: MagicMock) -> None:
    """Test process_output with multiple outputs containing both list and scalar values."""
    # Create multiple tool messages with different types of artifacts
    message1 = ToolMessage(
        tool_call_id="1",
        content="",
        artifact=LocalDataValue(value=["item1", "item2"], summary="First list output"),
    )
    message2 = ToolMessage(
        tool_call_id="2",
        content="",
        artifact=LocalDataValue(value="scalar_value", summary="Second scalar output"),
    )
    message3 = ToolMessage(
        tool_call_id="3",
        content="",
        artifact=LocalDataValue(value=["item3", "item4"], summary="Third list output"),
    )

    result = process_output(step, [message1, message2, message3], clarifications=[])

    assert isinstance(result, Output)
    # The list values should be extended, scalar values should be appended
    expected_values = ["item1", "item2", "scalar_value", "item3", "item4"]
    assert result.get_value() == expected_values
    # The final summary should use the last output's summary since it has one
    assert result.get_summary() == "Third list output"


def test_process_output_with_multiple_outputs_no_summaries(step: MagicMock) -> None:
    """Test process_output with multiple outputs where none have summaries."""
    message1 = ToolMessage(tool_call_id="1", content="", artifact=LocalDataValue(value=["a", "b"]))
    message2 = ToolMessage(tool_call_id="2", content="", artifact=LocalDataValue(value="c"))
    message3 = ToolMessage(tool_call_id="3", content="", artifact=LocalDataValue(value=["d", "e"]))

    result = process_output(step, [message1, message2, message3], clarifications=[])

    assert isinstance(result, Output)
    expected_values = ["a", "b", "c", "d", "e"]
    assert result.get_value() == expected_values
    # When no summaries are provided, it should join the serialized values
    expected_summary = '["a", "b"], c, ["d", "e"]'
    assert result.get_summary() == expected_summary


def test_process_output_with_nested_list_values(step: MagicMock) -> None:
    """Test process_output with outputs containing nested list structures."""
    message1 = ToolMessage(
        tool_call_id="1",
        content="",
        artifact=LocalDataValue(value=[{"id": 1}, {"id": 2}], summary="User list"),
    )
    message2 = ToolMessage(
        tool_call_id="2",
        content="",
        artifact=LocalDataValue(value={"status": "success"}, summary="Status update"),
    )
    message3 = ToolMessage(
        tool_call_id="3",
        content="",
        artifact=LocalDataValue(value=[{"id": 3}, {"id": 4}], summary="Final user list"),
    )

    result = process_output(step, [message1, message2, message3], clarifications=[])

    assert isinstance(result, Output)
    expected_values = [{"id": 1}, {"id": 2}, {"status": "success"}, {"id": 3}, {"id": 4}]
    assert result.get_value() == expected_values
    # Should use the last output's summary
    assert result.get_summary() == "Final user list"


def test_process_output_with_structured_output_schema(step: MagicMock) -> None:
    """Test process_output with structured output schema returns only the last tool call's value."""
    # Set up step with structured output schema
    step.structured_output_schema = {"type": "object", "properties": {"result": {"type": "string"}}}

    # Create multiple tool messages
    message1 = ToolMessage(
        tool_call_id="1",
        content="",
        artifact=LocalDataValue(value={"intermediate": "data1"}, summary="First output"),
    )
    message2 = ToolMessage(
        tool_call_id="2",
        content="",
        artifact=LocalDataValue(value={"intermediate": "data2"}, summary="Second output"),
    )
    message3 = ToolMessage(
        tool_call_id="3",
        content="",
        artifact=LocalDataValue(
            value={"result": "final_structured_output"}, summary="Final structured output"
        ),
    )

    result = process_output(step, [message1, message2, message3], clarifications=[])

    assert isinstance(result, Output)
    # When structured_output_schema is present, should return only the last tool call's value
    assert result.get_value() == {"result": "final_structured_output"}
    assert result.get_summary() == "Final structured output"


def test_process_output_with_structured_output_schema_no_summary(step: MagicMock) -> None:
    """Test process_output with structured output schema when last output has no summary."""
    # Set up step with structured output schema
    step.structured_output_schema = {"type": "object", "properties": {"result": {"type": "string"}}}

    # Create multiple tool messages to avoid single output case, with last one having no summary
    message1 = ToolMessage(
        tool_call_id="1",
        content="",
        artifact=LocalDataValue(value={"intermediate": "data"}, summary="First output"),
    )
    message2 = ToolMessage(
        tool_call_id="2",
        content="",
        artifact=LocalDataValue(value={"result": "structured_data"}),  # No summary provided
    )

    result = process_output(step, [message1, message2], clarifications=[])

    assert isinstance(result, Output)
    assert result.get_value() == {"result": "structured_data"}
    # Should use serialized value when no summary is provided
    assert result.get_summary() == '{"result": "structured_data"}'
