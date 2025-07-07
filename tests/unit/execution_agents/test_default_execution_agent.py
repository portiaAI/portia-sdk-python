"""test default execution agent."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from portia.clarification import InputClarification
from portia.end_user import EndUser
from portia.errors import InvalidAgentError, InvalidPlanRunStateError
from portia.execution_agents.context import StepInput
from portia.execution_agents.default_execution_agent import (
    MAX_RETRIES,
    DefaultExecutionAgent,
    ParserModel,
    ToolArgument,
    ToolCallingModel,
    ToolInputs,
    VerifiedToolArgument,
    VerifiedToolInputs,
    VerifierModel,
)
from portia.execution_agents.memory_extraction import MemoryExtractionStep
from portia.execution_agents.output import LocalDataValue, Output
from portia.execution_hooks import ExecutionHooks
from portia.model import LangChainGenerativeModel
from portia.plan import ReadOnlyStep, Step, Variable
from portia.plan_run import ReadOnlyPlanRun
from portia.storage import InMemoryStorage
from portia.tool import Tool
from tests.utils import (
    AdditionTool,
    get_mock_base_chat_model,
    get_mock_generative_model,
    get_test_config,
    get_test_plan_run,
    get_test_tool_context,
)


@pytest.fixture(scope="session", autouse=True)
def _setup() -> None:
    logging.basicConfig(level=logging.INFO)


class _TestToolSchema(BaseModel):
    """Input for TestTool."""

    content: str = Field(..., description="INPUT_DESCRIPTION")


def test_parser_model() -> None:
    """Test the parser model."""
    tool_inputs = ToolInputs(
        args=[
            ToolArgument(
                name="content",
                value="CONTENT_STRING",
                valid=True,
                explanation="EXPLANATION_STRING",
            ),
        ],
    )
    mock_model = get_mock_base_chat_model(response=tool_inputs)

    agent = SimpleNamespace(
        step=Step(task="DESCRIPTION_STRING", output="$out"),
        tool=SimpleNamespace(
            id="TOOL_ID",
            name="TOOL_NAME",
            args_json_schema=_TestToolSchema.model_json_schema,
            args_schema=_TestToolSchema,
            description="TOOL_DESCRIPTION",
        ),
        new_clarifications=[],
    )
    agent.get_system_context = mock.MagicMock(return_value="CONTEXT_STRING")

    parser_model = ParserModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
        tool_context=get_test_tool_context(),
    )
    parser_model.invoke({"messages": [], "step_inputs": []})

    assert mock_model.invoke.called
    messages = mock_model.invoke.call_args[0][0]
    assert messages
    assert "You are a highly capable assistant" in messages[0].content  # type: ignore  # noqa: PGH003
    assert "CONTEXT_STRING" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "DESCRIPTION_STRING" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_NAME" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_DESCRIPTION" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "INPUT_DESCRIPTION" in messages[1].content  # type: ignore  # noqa: PGH003
    assert mock_model.with_structured_output.called
    assert mock_model.with_structured_output.call_args[0][0] == ToolInputs


def test_parser_model_with_retries() -> None:
    """Test the parser model with retries."""
    tool_inputs = ToolInputs(
        args=[],
    )
    mock_invoker = get_mock_base_chat_model(response=tool_inputs)

    agent = SimpleNamespace(
        step=Step(task="DESCRIPTION_STRING", output="$out"),
        tool=SimpleNamespace(
            id="TOOL_ID",
            name="TOOL_NAME",
            args_json_schema=_TestToolSchema.model_json_schema,
            args_schema=_TestToolSchema,
            description="TOOL_DESCRIPTION",
        ),
        new_clarifications=[],
    )
    agent.get_system_context = mock.MagicMock(return_value="CONTEXT_STRING")

    parser_model = ParserModel(
        model=LangChainGenerativeModel(client=mock_invoker, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
        tool_context=get_test_tool_context(),
    )

    with mock.patch.object(parser_model, "invoke", side_effect=parser_model.invoke) as mock_invoke:
        parser_model.invoke({"step_inputs": []})  # type: ignore  # noqa: PGH003

    assert mock_invoke.call_count == MAX_RETRIES + 1


def test_parser_model_with_retries_invalid_structured_response() -> None:
    """Test the parser model handling of invalid JSON and retries."""
    mock_model = get_mock_base_chat_model(
        response="NOT_A_PYDANTIC_MODEL_INSTANCE",
    )

    agent = SimpleNamespace(
        step=Step(task="DESCRIPTION_STRING", output="$out"),
        tool=SimpleNamespace(
            id="TOOL_ID",
            name="TOOL_NAME",
            args_json_schema=_TestToolSchema.model_json_schema,
            args_schema=_TestToolSchema,
            description="TOOL_DESCRIPTION",
        ),
        new_clarifications=[],
    )
    agent.get_system_context = mock.MagicMock(return_value="CONTEXT_STRING")

    parser_model = ParserModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
        tool_context=get_test_tool_context(),
    )

    with mock.patch.object(parser_model, "invoke", side_effect=parser_model.invoke) as mock_invoke:
        parser_model.invoke({"messages": [], "step_inputs": []})  # type: ignore  # noqa: PGH003

    assert mock_invoke.call_count == MAX_RETRIES + 1


def test_parser_model_with_invalid_args() -> None:
    """Test the parser model handling of invalid arguments and retries."""
    # First response contains one valid and one invalid argument
    invalid_tool_inputs = ToolInputs(
        args=[
            ToolArgument(
                name="content",
                value="VALID_CONTENT",
                valid=True,
                explanation="Valid content string",
            ),
            ToolArgument(
                name="number",
                value=42,
                valid=False,
                explanation="The number should be more than 42",
            ),
        ],
    )

    # Second response contains all valid arguments
    valid_tool_inputs = ToolInputs(
        args=[
            ToolArgument(
                name="content",
                value="VALID_CONTENT",
                valid=True,
                explanation="Valid content string",
            ),
            ToolArgument(
                name="number",
                value=43,
                valid=True,
                explanation="Valid number value",
            ),
        ],
    )

    responses = [invalid_tool_inputs, valid_tool_inputs]
    current_response_index = 0

    def mock_invoke(*_, **__):  # noqa: ANN002, ANN003, ANN202
        nonlocal current_response_index
        response = responses[current_response_index]
        current_response_index += 1
        return response

    mock_model = get_mock_base_chat_model(response=None)
    mock_model.invoke.side_effect = mock_invoke

    class TestSchema(BaseModel):
        content: str
        number: int

    agent = SimpleNamespace(
        step=Step(task="DESCRIPTION_STRING", output="$out"),
        tool=SimpleNamespace(
            id="TOOL_ID",
            name="TOOL_NAME",
            args_json_schema=TestSchema.model_json_schema,
            args_schema=TestSchema,
            description="TOOL_DESCRIPTION",
        ),
        new_clarifications=[],
    )
    agent.get_system_context = mock.MagicMock(return_value="CONTEXT_STRING")

    parser_model = ParserModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
        tool_context=get_test_tool_context(),
    )

    # First call should store the error and retry
    result = parser_model.invoke({"messages": [], "step_inputs": []})

    # Verify that the error was stored
    assert len(parser_model.previous_errors) == 1
    assert (
        parser_model.previous_errors[0]
        == "Error in argument number: The number should be more than 42\n"
    )

    # Verify that we got the valid response after retry
    result_inputs = ToolInputs.model_validate_json(result["messages"][0])
    assert len(result_inputs.args) == 2

    # Check both arguments in final result
    content_arg = next(arg for arg in result_inputs.args if arg.name == "content")
    number_arg = next(arg for arg in result_inputs.args if arg.name == "number")

    assert content_arg.valid
    assert content_arg.value == "VALID_CONTENT"
    assert number_arg.valid
    assert number_arg.value == 43


def test_parser_model_schema_validation_success_with_templating() -> None:
    """Test that schema validation is skipped when template variables are present."""

    class ComplexSchema(BaseModel):
        """Schema with complex types to test template variables."""

        email_list: list[str] = Field(..., description="List of email addresses")
        config_dict: dict[str, str] = Field(..., description="Configuration dictionary")

    tool_inputs = ToolInputs(
        args=[
            ToolArgument(
                name="email_list",
                value=[
                    "{{$user_email}}",
                    "default@example.com",
                ],  # Template single email into a list
                valid=True,
                explanation="List containing user email from template variable",
            ),
            ToolArgument(
                name="config_dict",
                value={"config": "{{$user_config}}"},  # Template dictionary variable
                valid=True,
                explanation="Configuration dictionary from template variable",
            ),
        ],
    )
    mock_model = get_mock_base_chat_model(response=tool_inputs)

    agent = SimpleNamespace(
        step=Step(
            task="DESCRIPTION_STRING",
            inputs=[
                Variable(name="$user_email", description="User's email"),
                Variable(name="$user_config", description="User's configuration"),
            ],
            output="$out",
        ),
        tool=SimpleNamespace(
            id="TOOL_ID",
            name="TOOL_NAME",
            args_json_schema=ComplexSchema.model_json_schema,
            args_schema=ComplexSchema,
            description="TOOL_DESCRIPTION",
        ),
        new_clarifications=[],
    )
    agent.get_system_context = mock.MagicMock(return_value="CONTEXT_STRING")

    parser_model = ParserModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
        tool_context=get_test_tool_context(),
    )

    parser_model.invoke(
        {
            "messages": [],
            "step_inputs": [
                StepInput(name="$user_email", value="user@example.com", description="User's email"),
                StepInput(
                    name="$user_config",
                    value={"api_key": "abc123", "timeout": "30"},
                    description="User's configuration",
                ),
            ],
        }
    )

    assert len(parser_model.previous_errors) == 0
    assert parser_model.retries == 0


def test_parser_model_schema_validation_failure_with_templating() -> None:
    """Test that the parser model validates args schema and raises errors with invalid values."""

    class EmailSchema(BaseModel):
        email: str = Field(..., pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

    tool_inputs = ToolInputs(
        args=[
            ToolArgument(
                name="email",
                # Using a templated variable with invalid format
                value="{{$user_email_invalid}}",
                valid=True,
                explanation="Invalid email format even with templating",
            ),
        ],
    )
    mock_model = get_mock_base_chat_model(response=tool_inputs)

    agent = SimpleNamespace(
        step=Step(
            task="DESCRIPTION_STRING",
            inputs=[Variable(name="$user_email_invalid", description="User's invalid email")],
            output="$out",
        ),
        tool=SimpleNamespace(
            id="TOOL_ID",
            name="TOOL_NAME",
            args_json_schema=EmailSchema.model_json_schema,
            args_schema=EmailSchema,
            description="TOOL_DESCRIPTION",
        ),
        new_clarifications=[],
    )
    agent.get_system_context = mock.MagicMock(return_value="CONTEXT_STRING")

    parser_model = ParserModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
        tool_context=get_test_tool_context(),
    )

    # Provide step inputs with invalid email format
    parser_model.invoke(
        {
            "messages": [],
            "step_inputs": [
                StepInput(
                    name="$user_email_invalid", value="not-an-email", description="User's email"
                ),
            ],
        }
    )

    assert len(parser_model.previous_errors) == MAX_RETRIES + 1
    assert any("validation error" in error.lower() for error in parser_model.previous_errors)
    assert parser_model.retries == MAX_RETRIES + 1


def test_verifier_model() -> None:
    """Test the verifier model."""
    tool_inputs = ToolInputs(
        args=[
            ToolArgument(
                name="content",
                value="CONTENT_STRING",
                valid=True,
                explanation="EXPLANATION_STRING",
            ),
        ],
    )
    verified_tool_inputs = VerifiedToolInputs(
        args=[VerifiedToolArgument(name="content", value="CONTENT_STRING", made_up=False)],
    )
    mock_model = get_mock_base_chat_model(response=verified_tool_inputs)

    agent = SimpleNamespace(
        step=Step(task="DESCRIPTION_STRING", output="$out"),
        tool=SimpleNamespace(
            id="TOOL_ID",
            name="TOOL_NAME",
            args_schema=_TestToolSchema,
            description="TOOL_DESCRIPTION",
            args_json_schema=_TestToolSchema.model_json_schema,
        ),
        new_clarifications=[],
    )
    agent.get_system_context = mock.MagicMock(return_value="CONTEXT_STRING")
    verifier_model = VerifierModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
        tool_context=get_test_tool_context(),
    )
    verifier_model.invoke(
        {
            "messages": [AIMessage(content=tool_inputs.model_dump_json(indent=2))],
            "step_inputs": [],
        },
    )

    assert mock_model.invoke.called  # type: ignore[reportFunctionMemberAccess]
    messages = mock_model.invoke.call_args[0][0]  # type: ignore[reportFunctionMemberAccess]
    assert "You are an expert reviewer" in messages[0].content  # type: ignore  # noqa: PGH003
    assert "CONTEXT_STRING" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "DESCRIPTION_STRING" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_NAME" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_DESCRIPTION" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "INPUT_DESCRIPTION" in messages[1].content  # type: ignore  # noqa: PGH003
    assert mock_model.with_structured_output.called
    assert mock_model.with_structured_output.call_args[0][0] == VerifiedToolInputs


def test_verifier_model_schema_validation() -> None:
    """Test the verifier model schema validation."""

    class TestSchema(BaseModel):
        required_field1: str
        required_field2: int
        optional_field: str | None = None

    verified_tool_inputs = VerifiedToolInputs(
        args=[
            VerifiedToolArgument(name="required_field1", value=None, schema_invalid=True),
            VerifiedToolArgument(name="required_field2", value=None, schema_invalid=True),
            VerifiedToolArgument(name="optional_field", value=None, schema_invalid=False),
        ],
    )
    mock_model = get_mock_base_chat_model(response=verified_tool_inputs)

    agent = SimpleNamespace(
        step=Step(task="DESCRIPTION_STRING", output="$out"),
        tool=SimpleNamespace(
            id="TOOL_ID",
            name="TOOL_NAME",
            args_schema=TestSchema,
            description="TOOL_DESCRIPTION",
            args_json_schema=_TestToolSchema.model_json_schema,
        ),
        new_clarifications=[],
    )
    agent.get_system_context = mock.MagicMock(return_value="CONTEXT_STRING")
    verifier_model = VerifierModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
        tool_context=get_test_tool_context(),
    )

    result = verifier_model.invoke(
        {
            "messages": [AIMessage(content=verified_tool_inputs.model_dump_json(indent=2))],
            "step_inputs": [],
        },
    )

    result_inputs = VerifiedToolInputs.model_validate_json(result["messages"][0])

    required_field1 = next(arg for arg in result_inputs.args if arg.name == "required_field1")
    required_field2 = next(arg for arg in result_inputs.args if arg.name == "required_field2")
    assert (
        required_field1.schema_invalid
    ), "required_field1 should be marked as missing when validation fails"
    assert (
        required_field2.schema_invalid
    ), "required_field2 should be marked as missing when validation fails"

    optional_field = next(arg for arg in result_inputs.args if arg.name == "optional_field")
    assert (
        not optional_field.schema_invalid
    ), "optional_field should not be marked as missing when validation fails"


def test_verifier_model_validates_schema_with_templating() -> None:
    """Test that verifier model validates arguments against schema."""

    class EmailSchema(BaseModel):
        email: str = Field(..., pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

    verified_tool_inputs = VerifiedToolInputs(
        args=[
            VerifiedToolArgument(name="email", value="{{$invalid_email}}", made_up=False),
        ],
    )
    mock_model = get_mock_base_chat_model(response=verified_tool_inputs)

    agent = SimpleNamespace(
        step=Step(
            task="DESCRIPTION_STRING",
            inputs=[Variable(name="$invalid_email", description="User's email that is invalid")],
            output="$out",
        ),
        tool=SimpleNamespace(
            id="TOOL_ID",
            name="TOOL_NAME",
            args_schema=EmailSchema,
            args_json_schema=EmailSchema.model_json_schema,
            description="TOOL_DESCRIPTION",
        ),
        new_clarifications=[],
    )
    agent.get_system_context = mock.MagicMock(return_value="CONTEXT_STRING")

    verifier_model = VerifierModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
        tool_context=get_test_tool_context(),
    )

    result = verifier_model.invoke(
        {
            "messages": [AIMessage(content=verified_tool_inputs.model_dump_json(indent=2))],
            "step_inputs": [
                StepInput(
                    name="$invalid_email", value="not-valid@email", description="User's email"
                ),
            ],
        },
    )

    output_args = VerifiedToolInputs.model_validate_json(result["messages"][0])
    assert output_args.args[0].schema_invalid is True


def test_tool_calling_model_no_hallucinations() -> None:
    """Test the tool calling model."""
    verified_tool_inputs = VerifiedToolInputs(
        args=[VerifiedToolArgument(name="content", value="CONTENT_STRING", made_up=False)],
    )
    mock_model = get_mock_generative_model(
        SimpleNamespace(
            tool_calls=[
                {
                    "name": "add_tool",
                    "args": {
                        "arg1": "value1",
                    },
                }
            ]
        ),
    )

    (_, plan_run) = get_test_plan_run()
    mock_before_tool_call = mock.MagicMock(return_value=None)
    mock_telemetry = mock.MagicMock()
    agent = SimpleNamespace(
        plan_run=plan_run,
        step=Step(task="DESCRIPTION_STRING", output="$out"),
        tool=SimpleNamespace(
            id="TOOL_ID",
            name="TOOL_NAME",
            args_json_schema=_TestToolSchema,
            description="TOOL_DESCRIPTION",
        ),
        verified_args=verified_tool_inputs,
        clarifications=[],
        execution_hooks=ExecutionHooks(
            before_tool_call=mock_before_tool_call,
        ),
        new_clarifications=[],
        telemetry=mock_telemetry,
    )
    tool_calling_model = ToolCallingModel(
        model=mock_model,
        tools=[AdditionTool().to_langchain_with_artifact(ctx=get_test_tool_context())],
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    tool_calling_model.invoke({"messages": [], "step_inputs": []})

    base_chat_model = mock_model.to_langchain()
    assert base_chat_model.invoke.called  # type: ignore[reportFunctionMemberAccess]
    messages = base_chat_model.invoke.call_args[0][0]  # type: ignore[reportFunctionMemberAccess]
    assert "You are very powerful assistant" in messages[0].content  # type: ignore  # noqa: PGH003
    assert "DESCRIPTION_STRING" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_NAME" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "INPUT_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003
    mock_before_tool_call.assert_called_once_with(
        agent.tool,
        {"arg1": "value1"},
        ReadOnlyPlanRun.from_plan_run(agent.plan_run),
        ReadOnlyStep.from_step(agent.step),
    )
    # Verify telemetry was captured
    mock_telemetry.capture.assert_called_once()
    telemetry_call = mock_telemetry.capture.call_args[0][0]
    assert telemetry_call.tool_id == "TOOL_ID"


def test_tool_calling_model_with_hallucinations() -> None:
    """Test the tool calling model."""
    verified_tool_inputs = VerifiedToolInputs(
        args=[VerifiedToolArgument(name="content", value="CONTENT_STRING", made_up=True)],
    )
    mock_model = get_mock_generative_model(
        SimpleNamespace(
            tool_calls=[
                {
                    "name": "add_tool",
                    "args": {
                        "arg1": "value1",
                    },
                }
            ]
        ),
    )

    (_, plan_run) = get_test_plan_run()
    mock_telemetry = mock.MagicMock()

    clarification = InputClarification(
        plan_run_id=plan_run.id,
        user_guidance="USER_GUIDANCE",
        response="CLARIFICATION_RESPONSE",
        argument_name="content",
        resolved=True,
        source="Test tool calling model with hallucinations",
    )

    failed_clarification = InputClarification(
        plan_run_id=plan_run.id,
        user_guidance="USER_GUIDANCE_FAILED",
        response="FAILED",
        argument_name="content",
        resolved=True,
        source="Test tool calling model with hallucinations",
    )

    plan_run.outputs.clarifications = [clarification]
    agent = SimpleNamespace(
        verified_args=verified_tool_inputs,
        clarifications=[failed_clarification, clarification],
        missing_args={"content": clarification},
        get_last_resolved_clarification=lambda arg_name: clarification
        if arg_name == "content"
        else None,
        new_clarifications=[],
        execution_hooks=None,
        telemetry=mock_telemetry,
    )
    agent.step = Step(task="DESCRIPTION_STRING", output="$out")
    agent.plan_run = plan_run
    agent.tool = SimpleNamespace(
        id="TOOL_ID",
        name="TOOL_NAME",
        args_json_schema=_TestToolSchema,
        description="TOOL_DESCRIPTION",
    )
    tool_calling_model = ToolCallingModel(
        model=mock_model,
        tools=[AdditionTool().to_langchain_with_artifact(ctx=get_test_tool_context())],
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    tool_calling_model.invoke({"messages": [], "step_inputs": []})

    base_chat_model = mock_model.to_langchain()
    assert base_chat_model.invoke.called  # type: ignore[reportFunctionMemberAccess]
    messages = base_chat_model.invoke.call_args[0][0]  # type: ignore[reportFunctionMemberAccess]
    assert "You are very powerful assistant" in messages[0].content  # type: ignore  # noqa: PGH003
    assert "DESCRIPTION_STRING" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_NAME" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "INPUT_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003
    # Verify telemetry was captured
    mock_telemetry.capture.assert_called_once()
    telemetry_call = mock_telemetry.capture.call_args[0][0]
    assert telemetry_call.tool_id == "TOOL_ID"


def test_tool_calling_model_templates_inputs() -> None:
    """Test that the tool calling model correctly templates inputs into arguments."""
    templated_response = AIMessage(content="")
    templated_response.tool_calls = [
        {
            "name": "test_tool",
            "type": "tool_call",
            "id": "call_123",
            "args": {
                # Both cases (with and without $ at the start) should be templated
                "$templated_arg": "1. {{$input_value}} 2. {{input_value}}",
                "$normal_arg": "normal value",  # This should remain unchanged
            },
        },
    ]

    (_, plan_run) = get_test_plan_run()
    mock_model = get_mock_generative_model(templated_response)
    mock_telemetry = mock.MagicMock()
    step_inputs = [
        StepInput(name="$input_value", value="templated value", description="Input value"),
    ]
    verified_tool_inputs = VerifiedToolInputs(
        args=[
            VerifiedToolArgument(name="$templated_arg", value="{{$input_value}}", made_up=False),
            VerifiedToolArgument(name="$normal_arg", value="normal value", made_up=False),
        ],
    )

    addition_tool = AdditionTool()
    agent = SimpleNamespace(
        verified_args=verified_tool_inputs,
        step=Step(
            task="TASK_STRING",
            inputs=[Variable(name="$input_value", description="Input value")],
            output="$out",
        ),
        new_clarifications=[],
        execution_hooks=None,
        telemetry=mock_telemetry,
        tool=addition_tool,
    )
    agent.plan_run = plan_run
    tool_calling_model = ToolCallingModel(
        model=mock_model,
        tools=[addition_tool.to_langchain_with_artifact(ctx=get_test_tool_context())],
        agent=agent,  # type: ignore  # noqa: PGH003
    )

    result = tool_calling_model.invoke({"messages": [], "step_inputs": step_inputs})

    result_message = result["messages"][0]
    assert len(result_message.tool_calls) == 1
    tool_call = result_message.tool_calls[0]

    assert tool_call["args"]["$templated_arg"] == "1. templated value 2. templated value"
    assert tool_call["args"]["$normal_arg"] == "normal value"

    # Verify telemetry was captured
    mock_telemetry.capture.assert_called_once()
    telemetry_call = mock_telemetry.capture.call_args[0][0]
    assert telemetry_call.tool_id == addition_tool.id


def test_tool_calling_model_handles_missing_args_gracefully() -> None:
    """Test that the tool calling model handles missing args gracefully."""
    invalid_response = AIMessage(content="")
    invalid_response.tool_calls = [  # pyright: ignore[reportAttributeAccessIssue]
        {
            "name": "test_tool",
            "type": "tool_call",
            "id": "call_123",
            # args field is missing
        },
    ]

    (_, plan_run) = get_test_plan_run()
    mock_model = get_mock_generative_model(invalid_response)
    mock_telemetry = mock.MagicMock()
    verified_tool_inputs = VerifiedToolInputs(
        args=[
            VerifiedToolArgument(name="arg1", value="value1", made_up=False),
        ],
    )
    addition_tool = AdditionTool()
    agent = SimpleNamespace(
        verified_args=verified_tool_inputs,
        step=Step(task="TASK_STRING", inputs=[], output="$out"),
        new_clarifications=[],
        telemetry=mock_telemetry,
        tool=addition_tool,
    )
    agent.plan_run = plan_run
    tool_calling_model = ToolCallingModel(
        model=mock_model,
        tools=[addition_tool.to_langchain_with_artifact(ctx=get_test_tool_context())],
        agent=agent,  # type: ignore  # noqa: PGH003
    )

    with pytest.raises(InvalidPlanRunStateError):
        tool_calling_model.invoke({"messages": [], "step_inputs": []})


def test_basic_agent_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running an agent without a tool.

    Note: This tests mocks almost everything, but allows us to make sure things
    are running in order and being called correctly and passed out correctly.
    """
    tool_inputs = ToolInputs(
        args=[
            ToolArgument(
                name="email_address",
                valid=True,
                value="test@example.com",
                explanation="It's an email address.",
            ),
        ],
    )
    verified_tool_inputs = VerifiedToolInputs(
        args=[
            VerifiedToolArgument(name="email_address", value="test@example.com", made_up=False),
        ],
    )

    tool = AdditionTool()

    def memory_extraction_step(self, state):  # noqa: ANN001, ANN202, ARG001
        return {"step_inputs": []}

    monkeypatch.setattr(MemoryExtractionStep, "invoke", memory_extraction_step)

    def parser_model(self, state):  # noqa: ANN001, ANN202, ARG001
        return {"messages": [tool_inputs.model_dump_json(indent=2)]}

    monkeypatch.setattr(ParserModel, "invoke", parser_model)

    def verifier_model(self, state):  # noqa: ANN001, ANN202, ARG001
        self.agent.verified_args = verified_tool_inputs
        return {"messages": [verified_tool_inputs.model_dump_json(indent=2)]}

    monkeypatch.setattr(VerifierModel, "invoke", verifier_model)

    def tool_calling_model(self, state):  # noqa: ANN001, ANN202, ARG001
        response = AIMessage(content="")
        response.tool_calls = [
            {
                "name": "add_tool",
                "type": "tool_call",
                "id": "call_3z9rYHY6Rui7rTW0O7N7Wz51",
                "args": {
                    "recipients": ["test@example.com"],
                    "email_title": "Hi",
                    "email_body": "Hi",
                },
            },
        ]
        return {"messages": [response]}

    monkeypatch.setattr(ToolCallingModel, "invoke", tool_calling_model)

    def tool_call(self, input, config):  # noqa: A002, ANN001, ANN202, ARG001
        return {
            "messages": ToolMessage(
                content="Sent email",
                artifact=LocalDataValue(value="Sent email with id: 0"),
                tool_call_id="call_3z9rYHY6Rui7rTW0O7N7Wz51",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    mock_after_tool_call = mock.MagicMock(return_value=None)
    (plan, plan_run) = get_test_plan_run()
    agent = DefaultExecutionAgent(
        plan=plan,
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(),
        tool=tool,
        agent_memory=InMemoryStorage(),
        execution_hooks=ExecutionHooks(
            after_tool_call=mock_after_tool_call,
        ),
    )

    output = agent.execute_sync()
    assert isinstance(output, Output)
    assert output.get_value() == "Sent email with id: 0"
    mock_after_tool_call.assert_called_once_with(
        agent.tool,
        "Sent email",
        ReadOnlyPlanRun.from_plan_run(agent.plan_run),
        ReadOnlyStep.from_step(agent.step),
    )


def test_basic_agent_task_with_verified_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running an agent with verified args.

    Note: This tests mocks almost everything, but allows us to make sure things
    are running in order and being called correctly and passed out correctly.
    """
    verified_tool_inputs = VerifiedToolInputs(
        args=[
            VerifiedToolArgument(name="email_address", value="test@example.com", made_up=False),
        ],
    )

    tool = AdditionTool()

    def tool_calling_model(self, state):  # noqa: ANN001, ANN202, ARG001
        response = AIMessage(content="")
        response.tool_calls = [
            {
                "name": "add_tool",
                "type": "tool_call",
                "id": "call_3z9rYHY6Rui7rTW0O7N7Wz51",
                "args": {
                    "recipients": ["test@example.com"],
                    "email_title": "Hi",
                    "email_body": "Hi",
                },
            },
        ]
        return {"messages": [response]}

    monkeypatch.setattr(ToolCallingModel, "invoke", tool_calling_model)

    def tool_call(self, input, config):  # noqa: A002, ANN001, ANN202, ARG001
        return {
            "messages": ToolMessage(
                content="Sent email",
                artifact=LocalDataValue(value="Sent email with id: 0"),
                tool_call_id="call_3z9rYHY6Rui7rTW0O7N7Wz51",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    (plan, plan_run) = get_test_plan_run()
    agent = DefaultExecutionAgent(
        plan=plan,
        plan_run=plan_run,
        config=get_test_config(),
        end_user=EndUser(external_id="123"),
        tool=tool,
        agent_memory=InMemoryStorage(),
    )
    agent.verified_args = verified_tool_inputs

    output = agent.execute_sync()
    assert isinstance(output, Output)
    assert output.get_value() == "Sent email with id: 0"


def test_default_execution_agent_edge_cases() -> None:
    """Tests edge cases are handled."""
    agent = SimpleNamespace(
        step=Step(task="DESCRIPTION_STRING", output="$out"),
        tool=None,
        verified_args=None,
        new_clarifications=[],
    )
    parser_model = ParserModel(
        model=get_mock_generative_model(get_mock_base_chat_model()),
        agent=agent,  # type: ignore  # noqa: PGH003
        tool_context=get_test_tool_context(),
    )
    with pytest.raises(InvalidPlanRunStateError):
        parser_model.invoke({"messages": [], "step_inputs": []})

    tool_calling_model = ToolCallingModel(
        model=get_mock_generative_model(get_mock_base_chat_model()),
        tools=[AdditionTool().to_langchain_with_artifact(ctx=get_test_tool_context())],
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    with pytest.raises(InvalidPlanRunStateError):
        tool_calling_model.invoke({"messages": [], "step_inputs": []})


def test_get_last_resolved_clarification() -> None:
    """Test get_last_resolved_clarification."""
    (plan, plan_run) = get_test_plan_run()
    resolved_clarification1 = InputClarification(
        plan_run_id=plan_run.id,
        argument_name="arg",
        response="2",
        user_guidance="FAILED",
        resolved=True,
        step=0,
        source="Test get last resolved clarification",
    )
    resolved_clarification2 = InputClarification(
        plan_run_id=plan_run.id,
        argument_name="arg",
        response="2",
        user_guidance="SUCCESS",
        resolved=True,
        step=0,
        source="Test get last resolved clarification",
    )
    unresolved_clarification = InputClarification(
        plan_run_id=plan_run.id,
        argument_name="arg",
        response="2",
        user_guidance="",
        resolved=False,
        step=0,
        source="Test get last resolved clarification",
    )
    plan_run.outputs.clarifications = [
        resolved_clarification1,
        resolved_clarification2,
        unresolved_clarification,
    ]
    agent = DefaultExecutionAgent(
        plan=plan,
        plan_run=plan_run,
        config=get_test_config(),
        end_user=EndUser(external_id="123"),
        tool=None,
        agent_memory=InMemoryStorage(),
    )
    assert agent.get_last_resolved_clarification("arg") == resolved_clarification2


def test_clarifications_or_continue() -> None:
    """Test clarifications_or_continue."""
    (plan, plan_run) = get_test_plan_run()
    clarification = InputClarification(
        plan_run_id=plan_run.id,
        argument_name="arg",
        response="2",
        user_guidance="",
        resolved=True,
        source="Test clarifications or continue",
    )

    agent = DefaultExecutionAgent(
        plan=plan,
        plan_run=plan_run,
        config=get_test_config(),
        end_user=EndUser(external_id="123"),
        tool=None,
        agent_memory=InMemoryStorage(),
    )
    inputs = VerifiedToolInputs(
        args=[
            VerifiedToolArgument(name="arg", value="1", made_up=True),
        ],
    )

    # when clarifications don't match expect a new one
    output = agent.clarifications_or_continue(
        {
            "messages": [
                HumanMessage(
                    content=inputs.model_dump_json(indent=2),
                ),
            ],
            "step_inputs": [],
        },
    )
    assert output == END
    assert isinstance(agent.new_clarifications, list)
    assert isinstance(agent.new_clarifications[0], InputClarification)

    # when clarifications match expect to call tools
    clarification = InputClarification(
        plan_run_id=plan_run.id,
        argument_name="arg",
        response="1",
        user_guidance="",
        resolved=True,
        step=0,
        source="Test clarifications or continue",
    )

    (plan, plan_run) = get_test_plan_run()
    plan_run.outputs.clarifications = [clarification]
    agent = DefaultExecutionAgent(
        plan=plan,
        end_user=EndUser(external_id="123"),
        plan_run=plan_run,
        config=get_test_config(),
        tool=None,
        agent_memory=InMemoryStorage(),
    )

    inputs = VerifiedToolInputs(
        args=[
            VerifiedToolArgument(name="arg", value="1", made_up=True),
        ],
    )

    output = agent.clarifications_or_continue(
        {
            "messages": [
                HumanMessage(
                    content=inputs.model_dump_json(indent=2),
                ),
            ],
            "step_inputs": [],
        },
    )
    assert output == "tool_agent"
    assert isinstance(agent.new_clarifications, list)
    assert len(agent.new_clarifications) == 0


def test_default_execution_agent_none_tool_execute_sync() -> None:
    """Test that executing DefaultExecutionAgent with None tool raises an exception."""
    (plan, plan_run) = get_test_plan_run()

    agent = DefaultExecutionAgent(
        plan=plan,
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(),
        tool=None,
        agent_memory=InMemoryStorage(),
    )

    with pytest.raises(InvalidAgentError) as exc_info:
        agent.execute_sync()

    assert "Tool is required for DefaultExecutionAgent" in str(exc_info.value)


class MockToolSchema(BaseModel):
    """Mock tool schema."""

    optional_arg: str | None = Field(default=None, description="An optional argument")


class MockAgent:
    """Mock agent."""

    def __init__(self) -> None:
        """Init mock agent."""
        self.tool = MockTool()


class MockTool(Tool):
    """Mock tool."""

    def __init__(self) -> None:
        """Init mock tool."""
        super().__init__(
            name="Mock Tool",
            id="mock_tool",
            description="Mock tool description",
            args_schema=MockToolSchema,
            output_schema=("type", "A description of the output"),
        )

    def run(self, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002
        """Run mock tool."""
        return "RUN_RESULT"


def test_optional_args_with_none_values() -> None:
    """Test that optional args with None values are handled correctly.

    Required args with None values should always be marked made_up.
    Optional args with None values should be marked not made_up.
    """
    (plan, plan_run) = get_test_plan_run()
    agent = DefaultExecutionAgent(
        plan=plan,
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(),
        tool=MockTool(),
        agent_memory=InMemoryStorage(),
    )
    model = VerifierModel(
        model=LangChainGenerativeModel(client=get_mock_base_chat_model(), model_name="test"),
        agent=agent,
        tool_context=get_test_tool_context(),
    )

    #  Optional arg and made_up is True == not made_up
    updated_tool_inputs = model._validate_args_against_schema(
        VerifiedToolInputs(
            args=[VerifiedToolArgument(name="optional_arg", value=None, made_up=True)],
        ),
        [],
    )
    assert updated_tool_inputs.args[0].made_up is False

    #  Optional arg and made_up is False == mnot ade_up
    updated_tool_inputs = model._validate_args_against_schema(
        VerifiedToolInputs(
            args=[VerifiedToolArgument(name="optional_arg", value=None, made_up=False)],
        ),
        [],
    )
    assert updated_tool_inputs.args[0].made_up is False


class ListToolSchema(BaseModel):
    """List tool schema."""

    list_arg: list[str] = Field(..., description="An optional argument")


class ListToolAgent:
    """ListTool agent."""

    def __init__(self) -> None:
        """Init mock agent."""
        self.tool = ListTool()


class ListTool(Tool):
    """List tool."""

    def __init__(self) -> None:
        """Init List tool."""
        super().__init__(
            name="List Tool",
            id="list_tool",
            description="List tool description",
            args_schema=ListToolSchema,
            output_schema=("type", "A description of the output"),
        )

    def run(self, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002
        """Run mock tool."""
        return "RUN_RESULT"


def test_list_args_with_str_values() -> None:
    """Test that list args with str values are handled correctly."""
    (plan, plan_run) = get_test_plan_run()
    agent = DefaultExecutionAgent(
        plan=plan,
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(),
        tool=ListTool(),
        agent_memory=InMemoryStorage(),
    )
    model = VerifierModel(
        model=LangChainGenerativeModel(client=get_mock_base_chat_model(), model_name="test"),
        agent=agent,
        tool_context=get_test_tool_context(),
    )

    updated_tool_inputs = model._validate_args_against_schema(
        VerifiedToolInputs(
            args=[VerifiedToolArgument(name="list_arg", value="[1,2,3]", made_up=False)],
        ),
        [],
    )
    assert updated_tool_inputs.args[0].schema_invalid is False


def test_verifier_model_edge_cases() -> None:
    """Tests edge cases are handled."""
    agent = SimpleNamespace(
        step=Step(task="DESCRIPTION_STRING", output="$out"),
        tool=None,
        new_clarifications=[],
    )
    verifier_model = VerifierModel(
        model=LangChainGenerativeModel(client=get_mock_base_chat_model(), model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
        tool_context=get_test_tool_context(),
    )

    # Check error with no tool specified
    with pytest.raises(InvalidPlanRunStateError):
        verifier_model.invoke({"messages": [], "step_inputs": []})


def test_before_tool_call_with_clarification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that before_tool_call can interrupt execution by returning a clarification."""
    model_response = AIMessage(content="")
    model_response.tool_calls = [
        {
            "name": "Send_Email_Tool",
            "type": "tool_call",
            "id": "call_3z9rYHY6Rui7rTW0O7N7Wz51",
            "args": {
                "recipients": ["test@example.com"],
                "email_title": "Hi",
                "email_body": "Hi",
            },
        },
    ]
    mock_model = get_mock_generative_model(response=model_response)
    monkeypatch.setattr("portia.config.Config.get_execution_model", lambda self: mock_model)  # noqa: ARG005

    tool_node_called = False

    def tool_call(self, input, config):  # noqa: A002, ANN001, ANN202, ARG001
        nonlocal tool_node_called
        tool_node_called = True
        return {
            "messages": ToolMessage(
                content="Added numbers",
                artifact=LocalDataValue(value=3),
                tool_call_id="call_3z9rYHY6Rui7rTW0O7N7Wz51",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    return_clarification = True

    def before_tool_call(tool, args, plan_run, step) -> InputClarification | None:  # noqa: ANN001, ARG001
        nonlocal return_clarification
        if return_clarification:
            return InputClarification(
                plan_run_id=plan_run.id,
                user_guidance="Need clarification",
                step=plan_run.current_step_index,
                argument_name="num1",
                source="Test before tool call with clarification",
            )
        return None

    (plan, plan_run) = get_test_plan_run()

    # First execution - should return clarification
    agent = DefaultExecutionAgent(
        plan=plan,
        plan_run=plan_run,
        config=get_test_config(),
        end_user=EndUser(external_id="123"),
        tool=AdditionTool(),
        agent_memory=InMemoryStorage(),
        execution_hooks=ExecutionHooks(
            before_tool_call=before_tool_call,
        ),
    )
    agent.verified_args = VerifiedToolInputs(
        args=[
            VerifiedToolArgument(name="email_address", value="test@example.com", made_up=False),
        ],
    )
    output = agent.execute_sync()

    assert tool_node_called is False
    assert len(output.get_value()) == 1  # pyright: ignore[reportArgumentType]
    output_value = output.get_value()[0]  # pyright: ignore[reportOptionalSubscript]
    assert isinstance(output_value, InputClarification)
    assert output_value.user_guidance == "Need clarification"

    # Second execution - should call the tool
    return_clarification = False
    tool_node_called = False
    agent.new_clarifications = []
    output = agent.execute_sync()

    assert tool_node_called is True
    assert output.get_value() == 3


def test_after_tool_call_with_clarification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that after_tool_call can interrupt execution by returning a clarification."""
    model_response = AIMessage(content="")
    model_response.tool_calls = [
        {
            "name": "Send_Email_Tool",
            "type": "tool_call",
            "id": "call_3z9rYHY6Rui7rTW0O7N7Wz51",
            "args": {
                "recipients": ["test@example.com"],
                "email_title": "Hi",
                "email_body": "Hi",
            },
        },
    ]
    mock_model = get_mock_generative_model(response=model_response)
    monkeypatch.setattr("portia.config.Config.get_execution_model", lambda self: mock_model)  # noqa: ARG005

    tool_node_called = False

    def tool_call(self, input, config):  # noqa: A002, ANN001, ANN202, ARG001
        nonlocal tool_node_called
        tool_node_called = True
        return {
            "messages": ToolMessage(
                content="Added numbers",
                artifact=LocalDataValue(value=3),
                tool_call_id="call_3z9rYHY6Rui7rTW0O7N7Wz51",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    return_clarification = True

    def after_tool_call(tool, output, plan_run, step) -> InputClarification | None:  # noqa: ANN001, ARG001
        nonlocal return_clarification
        if return_clarification:
            return InputClarification(
                plan_run_id=plan_run.id,
                user_guidance="Need clarification after tool call",
                step=plan_run.current_step_index,
                argument_name="result",
                source="Test after tool call with clarification",
            )
        return None

    (plan, plan_run) = get_test_plan_run()

    # First execution - should return clarification after tool call
    agent = DefaultExecutionAgent(
        plan=plan,
        plan_run=plan_run,
        config=get_test_config(),
        end_user=EndUser(external_id="123"),
        tool=AdditionTool(),
        agent_memory=InMemoryStorage(),
        execution_hooks=ExecutionHooks(
            after_tool_call=after_tool_call,
        ),
    )
    agent.verified_args = VerifiedToolInputs(
        args=[
            VerifiedToolArgument(name="email_address", value="test@example.com", made_up=False),
        ],
    )
    output = agent.execute_sync()

    assert tool_node_called is True
    assert len(output.get_value()) == 1  # pyright: ignore[reportArgumentType]
    output_value = output.get_value()[0]  # pyright: ignore[reportOptionalSubscript]
    assert isinstance(output_value, InputClarification)
    assert output_value.user_guidance == "Need clarification after tool call"

    # Second execution - should complete without clarification
    return_clarification = False
    tool_node_called = False
    agent.new_clarifications = []
    output = agent.execute_sync()

    assert tool_node_called is True
    assert output.get_value() == 3
