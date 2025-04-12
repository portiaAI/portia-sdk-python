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
from portia.errors import InvalidAgentError, InvalidPlanRunStateError
from portia.execution_agents.default_execution_agent import (
    MAX_RETRIES,
    DefaultExecutionAgent,
    MemoryExtractionModel,
    MemoryExtractionModelOutput,
    ParserModel,
    ToolArgument,
    ToolCallingModel,
    ToolInputs,
    VerifiedToolArgument,
    VerifiedToolInputs,
    VerifierModel,
)
from portia.execution_agents.output import AgentMemoryOutput, LocalOutput, Output
from portia.model import LangChainGenerativeModel
from portia.plan import Step, Variable
from portia.storage import InMemoryStorage
from portia.tool import Tool
from tests.utils import (
    AdditionTool,
    get_mock_base_chat_model,
    get_mock_langchain_generative_model,
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
        contains_templated_argument=True,
    )
    mock_model = get_mock_base_chat_model(response=tool_inputs)

    agent = SimpleNamespace()
    agent.step = Step(task="DESCRIPTION_STRING", output="$out")
    agent.tool = SimpleNamespace(
        id="TOOL_ID",
        name="TOOL_NAME",
        args_json_schema=_TestToolSchema.model_json_schema,
        args_schema=_TestToolSchema,
        description="TOOL_DESCRIPTION",
    )
    parser_model = ParserModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
    )

    with mock.patch(
        "portia.execution_agents.default_execution_agent.get_execution_context",
        return_value="CONTEXT_STRING",
    ):
        result = parser_model.invoke({})  # type: ignore  # noqa: PGH003

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
    assert result["messages"][0] == tool_inputs.model_dump_json(indent=2)


def test_parser_model_with_retries() -> None:
    """Test the parser model with retries."""
    tool_inputs = ToolInputs(
        args=[],
    )
    mock_invoker = get_mock_base_chat_model(response=tool_inputs)

    agent = SimpleNamespace()
    agent.step = Step(task="DESCRIPTION_STRING", output="$out")
    agent.tool = SimpleNamespace(
        id="TOOL_ID",
        name="TOOL_NAME",
        args_json_schema=_TestToolSchema.model_json_schema,
        args_schema=_TestToolSchema,
        description="TOOL_DESCRIPTION",
    )
    parser_model = ParserModel(
        model=LangChainGenerativeModel(client=mock_invoker, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
    )

    with mock.patch.object(parser_model, "invoke", side_effect=parser_model.invoke) as mock_invoke:
        parser_model.invoke({})  # type: ignore  # noqa: PGH003

    assert mock_invoke.call_count == MAX_RETRIES + 1


def test_parser_model_with_retries_invalid_structured_response() -> None:
    """Test the parser model handling of invalid JSON and retries."""
    mock_model = get_mock_base_chat_model(
        response="NOT_A_PYDANTIC_MODEL_INSTANCE",
    )

    agent = SimpleNamespace()
    agent.step = Step(task="DESCRIPTION_STRING", output="$out")
    agent.tool = SimpleNamespace(
        id="TOOL_ID",
        name="TOOL_NAME",
        args_json_schema=_TestToolSchema.model_json_schema,
        args_schema=_TestToolSchema,
        description="TOOL_DESCRIPTION",
    )
    parser_model = ParserModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
    )

    with mock.patch.object(parser_model, "invoke", side_effect=parser_model.invoke) as mock_invoke:
        parser_model.invoke({"messages": []})  # type: ignore  # noqa: PGH003

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

    agent = SimpleNamespace()
    agent.step = Step(task="DESCRIPTION_STRING", output="$out")
    agent.tool = SimpleNamespace(
        id="TOOL_ID",
        name="TOOL_NAME",
        args_json_schema=TestSchema.model_json_schema,
        args_schema=TestSchema,
        description="TOOL_DESCRIPTION",
    )

    parser_model = ParserModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
    )

    # First call should store the error and retry
    result = parser_model.invoke({"messages": []})

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
        contains_templated_argument=True,
    )
    verified_tool_inputs = VerifiedToolInputs(
        args=[VerifiedToolArgument(name="content", value="CONTENT_STRING", made_up=False)],
    )
    mock_model = get_mock_base_chat_model(response=verified_tool_inputs)

    agent = SimpleNamespace()
    agent.step = Step(task="DESCRIPTION_STRING", output="$out")
    agent.tool = SimpleNamespace(
        id="TOOL_ID",
        name="TOOL_NAME",
        args_schema=_TestToolSchema,
        description="TOOL_DESCRIPTION",
        args_json_schema=_TestToolSchema.model_json_schema,
    )
    verifier_model = VerifierModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
    )

    with mock.patch(
        "portia.execution_agents.default_execution_agent.get_execution_context",
        return_value="CONTEXT_STRING",
    ):
        result = verifier_model.invoke(
            {"messages": [AIMessage(content=tool_inputs.model_dump_json(indent=2))]},
        )

    assert mock_model.invoke.called  # type: ignore[reportFunctionMemberAccess]
    messages = mock_model.invoke.call_args[0][0]  # type: ignore[reportFunctionMemberAccess]
    assert "You are an expert reviewer" in messages[0].content  # type: ignore  # noqa: PGH003
    assert "CONTEXT_STRING" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "DESCRIPTION_STRING" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_NAME" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "INPUT_DESCRIPTION" in messages[1].content  # type: ignore  # noqa: PGH003
    assert mock_model.with_structured_output.called
    assert mock_model.with_structured_output.call_args[0][0] == VerifiedToolInputs
    assert result["messages"][0] == verified_tool_inputs.model_dump_json(indent=2)


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

    agent = SimpleNamespace()
    agent.step = Step(task="DESCRIPTION_STRING", output="$out")
    agent.tool = SimpleNamespace(
        id="TOOL_ID",
        name="TOOL_NAME",
        args_schema=TestSchema,
        description="TOOL_DESCRIPTION",
        args_json_schema=_TestToolSchema.model_json_schema,
    )
    verifier_model = VerifierModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
    )

    result = verifier_model.invoke(
        {"messages": [AIMessage(content=verified_tool_inputs.model_dump_json(indent=2))]},
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


def test_tool_calling_model_no_hallucinations() -> None:
    """Test the tool calling model."""
    verified_tool_inputs = VerifiedToolInputs(
        args=[VerifiedToolArgument(name="content", value="CONTENT_STRING", made_up=False)],
    )
    args = {"arg1": "ARG1_VALUE", "arg2": "ARG2_VALUE"}
    mock_model = get_mock_langchain_generative_model(
        SimpleNamespace(content=[args]),
    )

    (_, plan_run) = get_test_plan_run()
    agent = SimpleNamespace(
        verified_args=verified_tool_inputs,
        clarifications=[],
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
    result = tool_calling_model.invoke({"messages": []})

    base_chat_model = mock_model.to_langchain()
    assert base_chat_model.invoke.called  # type: ignore[reportFunctionMemberAccess]
    messages = base_chat_model.invoke.call_args[0][0]  # type: ignore[reportFunctionMemberAccess]
    assert "You are very powerful assistant" in messages[0].content  # type: ignore  # noqa: PGH003
    assert "CONTEXT_STRING" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "DESCRIPTION_STRING" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_NAME" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "INPUT_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert result["messages"][0].content == [args]


def test_tool_calling_model_with_hallucinations() -> None:
    """Test the tool calling model."""
    verified_tool_inputs = VerifiedToolInputs(
        args=[VerifiedToolArgument(name="content", value="CONTENT_STRING", made_up=True)],
    )
    mock_model = get_mock_langchain_generative_model(
        SimpleNamespace(tool_calls=[{"name": "add_tool", "args": "CALL_ARGS"}]),
    )

    (_, plan_run) = get_test_plan_run()

    clarification = InputClarification(
        plan_run_id=plan_run.id,
        user_guidance="USER_GUIDANCE",
        response="CLARIFICATION_RESPONSE",
        argument_name="content",
        resolved=True,
    )

    failed_clarification = InputClarification(
        plan_run_id=plan_run.id,
        user_guidance="USER_GUIDANCE_FAILED",
        response="FAILED",
        argument_name="content",
        resolved=True,
    )

    plan_run.outputs.clarifications = [clarification]
    agent = SimpleNamespace(
        verified_args=verified_tool_inputs,
        clarifications=[failed_clarification, clarification],
        missing_args={"content": clarification},
        get_last_resolved_clarification=lambda arg_name: clarification
        if arg_name == "content"
        else None,
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
    tool_calling_model.invoke({"messages": []})

    base_chat_model = mock_model.to_langchain()
    assert base_chat_model.invoke.called  # type: ignore[reportFunctionMemberAccess]
    messages = base_chat_model.invoke.call_args[0][0]  # type: ignore[reportFunctionMemberAccess]
    assert "You are very powerful assistant" in messages[0].content  # type: ignore  # noqa: PGH003
    assert "CONTEXT_STRING" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "DESCRIPTION_STRING" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_NAME" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "INPUT_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003


def test_tool_calling_model_with_templated_args() -> None:
    """Test the tool calling model with templated arguments."""

    class TestSchema(BaseModel):
        """Test schema with two arguments."""

        recipient: str = Field(..., description="The recipient of the message")
        title: str = Field(..., description="The title of the message")
        body: str = Field(..., description="The body of the message")

    verified_tool_inputs = VerifiedToolInputs(
        args=[
            VerifiedToolArgument(name="recipient", value="test@example.com", made_up=False),
            VerifiedToolArgument(
                name="title",
                value="This is a long title: {{previous_output_1}}",
                made_up=False,
            ),
            VerifiedToolArgument(
                name="body",
                value="Hello, here's some info: {{previous_output_2}}",
                made_up=False,
            ),
        ],
        contains_templated_argument=True,
    )

    # Create a mock model that returns a response with templated values
    args = {
        "recipient": ["test@example.com"],
        "title": "This is a long title: {{previous_output_1}}",
        "body": "Hello, here's some info: {{previous_output_2}}",
    }
    mock_model = get_mock_langchain_generative_model(
        SimpleNamespace(content=str(args)),
    )

    (_, plan_run) = get_test_plan_run()
    plan_run.outputs.step_outputs = {
        "$previous_output_1": LocalOutput(value="Previous output 1"),
        "$previous_output_2": LocalOutput(value="Previous output 2"),
    }
    agent = SimpleNamespace(
        verified_args=verified_tool_inputs,
        clarifications=[],
        agent_memory=InMemoryStorage(),
    )
    agent.step = Step(task="Send an email to the recipient with the message", output="$out")
    agent.plan_run = plan_run
    agent.tool = SimpleNamespace(
        id="message_tool",
        name="TOOL_NAME",
        args_json_schema=_TestToolSchema,
        description="TOOL_DESCRIPTION",
    )

    tool_calling_model = ToolCallingModel(
        model=mock_model,
        tools=[AdditionTool().to_langchain_with_artifact(ctx=get_test_tool_context())],
        agent=agent,  # type: ignore  # noqa: PGH003
    )

    with mock.patch(
        "portia.execution_agents.default_execution_agent.get_execution_context",
        return_value="CONTEXT_STRING",
    ):
        result = tool_calling_model.invoke({"messages": []})

    # Verify the result contains the templated values
    assert "messages" in result
    assert len(result["messages"]) == 1
    message = result["messages"][0]
    assert "test@example.com" in message.content
    assert "This is a long title: Previous output 1" in message.content
    assert "Hello, here's some info: Previous output 2" in message.content
    # @@@ TODO: CHECK THE REAL OUTPUT FROM THE TOOL CALLING MODEL


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

    monkeypatch.setattr(MemoryExtractionModel, "invoke", lambda _1, _2: None)

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
                artifact=LocalOutput(value="Sent email with id: 0"),
                tool_call_id="call_3z9rYHY6Rui7rTW0O7N7Wz51",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    (plan, plan_run) = get_test_plan_run()
    agent = DefaultExecutionAgent(
        step=plan.steps[0],
        plan_run=plan_run,
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=tool,
    )

    output = agent.execute_sync()
    assert isinstance(output, Output)
    assert output.get_value() == "Sent email with id: 0"


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
                artifact=LocalOutput(value="Sent email with id: 0"),
                tool_call_id="call_3z9rYHY6Rui7rTW0O7N7Wz51",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    (plan, plan_run) = get_test_plan_run()
    agent = DefaultExecutionAgent(
        step=plan.steps[0],
        plan_run=plan_run,
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=tool,
    )
    agent.verified_args = verified_tool_inputs

    output = agent.execute_sync()
    assert isinstance(output, Output)
    assert output.get_value() == "Sent email with id: 0"


def test_default_execution_agent_edge_cases() -> None:
    """Tests edge cases are handled."""
    agent = SimpleNamespace()
    agent.step = Step(task="DESCRIPTION_STRING", output="$out")
    agent.tool = None
    parser_model = ParserModel(
        model=get_mock_langchain_generative_model(get_mock_base_chat_model()),
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    with pytest.raises(InvalidPlanRunStateError):
        parser_model.invoke({"messages": []})

    agent.verified_args = None
    tool_calling_model = ToolCallingModel(
        model=get_mock_langchain_generative_model(get_mock_base_chat_model()),
        tools=[AdditionTool().to_langchain_with_artifact(ctx=get_test_tool_context())],
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    with pytest.raises(InvalidPlanRunStateError):
        tool_calling_model.invoke({"messages": []})


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
    )
    resolved_clarification2 = InputClarification(
        plan_run_id=plan_run.id,
        argument_name="arg",
        response="2",
        user_guidance="SUCCESS",
        resolved=True,
        step=0,
    )
    unresolved_clarification = InputClarification(
        plan_run_id=plan_run.id,
        argument_name="arg",
        response="2",
        user_guidance="",
        resolved=False,
        step=0,
    )
    plan_run.outputs.clarifications = [
        resolved_clarification1,
        resolved_clarification2,
        unresolved_clarification,
    ]
    agent = DefaultExecutionAgent(
        step=plan.steps[0],
        plan_run=plan_run,
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=None,
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
    )

    agent = DefaultExecutionAgent(
        step=plan.steps[0],
        plan_run=plan_run,
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=None,
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
    )

    (plan, plan_run) = get_test_plan_run()
    plan_run.outputs.clarifications = [clarification]
    agent = DefaultExecutionAgent(
        step=plan.steps[0],
        plan_run=plan_run,
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=None,
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
        },
    )
    assert output == "tool_agent"
    assert isinstance(agent.new_clarifications, list)
    assert len(agent.new_clarifications) == 0


def test_default_execution_agent_none_tool_execute_sync() -> None:
    """Test that executing DefaultExecutionAgent with None tool raises an exception."""
    (plan, plan_run) = get_test_plan_run()

    agent = DefaultExecutionAgent(
        step=plan.steps[0],
        plan_run=plan_run,
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=None,
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
    agent = DefaultExecutionAgent(
        step=Step(task="TASK_STRING", output="$out"),
        plan_run=get_test_plan_run()[1],
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=MockTool(),
    )
    model = VerifierModel(
        model=LangChainGenerativeModel(client=get_mock_base_chat_model(), model_name="test"),
        agent=agent,
    )

    #  Optional arg and made_up is True == not made_up
    updated_tool_inputs = model._validate_args_against_schema(  # noqa: SLF001
        VerifiedToolInputs(
            args=[VerifiedToolArgument(name="optional_arg", value=None, made_up=True)],
        ),
    )
    assert updated_tool_inputs.args[0].made_up is False

    #  Optional arg and made_up is False == mnot ade_up
    updated_tool_inputs = model._validate_args_against_schema(  # noqa: SLF001
        VerifiedToolInputs(
            args=[VerifiedToolArgument(name="optional_arg", value=None, made_up=False)],
        ),
    )
    assert updated_tool_inputs.args[0].made_up is False


def test_verifier_model_edge_cases() -> None:
    """Tests edge cases are handled."""
    agent = SimpleNamespace()
    agent.step = Step(task="DESCRIPTION_STRING", output="$out")
    verifier_model = VerifierModel(
        model=LangChainGenerativeModel(client=get_mock_base_chat_model(), model_name="test"),
        agent=agent,  # type: ignore  # noqa: PGH003
    )

    # Check error with no tool specified
    agent.tool = None
    with pytest.raises(InvalidPlanRunStateError):
        verifier_model.invoke({"messages": []})


def test_memory_extraction_model_all_local_inputs() -> None:
    """Test the memory extraction model when all inputs are local."""
    mock_model = get_mock_base_chat_model(response=MemoryExtractionModelOutput())

    (_, plan_run) = get_test_plan_run()
    plan_run.outputs.step_outputs = {
        "$local_input1": LocalOutput(value="Local value 1"),
        "$local_input2": LocalOutput(value="Local value 2"),
    }

    agent = SimpleNamespace()
    agent.tool = SimpleNamespace(
        id="TOOL_ID",
        name="TOOL_NAME",
        args_json_schema=_TestToolSchema.model_json_schema,
        args_schema=_TestToolSchema,
        description="TOOL_DESCRIPTION",
    )
    agent.plan_run = plan_run
    agent.get_system_context = mock.Mock(return_value="MOCKED_SYSTEM_CONTEXT")

    memory_model = MemoryExtractionModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,
    )

    with mock.patch(
        "portia.execution_agents.default_execution_agent.get_execution_context",
    ) as mock_context:
        memory_model.invoke({})

        assert not mock_model.invoke.called
        assert mock_context.return_value.plan_run_context == "MOCKED_SYSTEM_CONTEXT"


def test_memory_extraction_model_mixed_inputs_no_extraction() -> None:
    """Test memory extraction with mixed inputs, but no extraction needed."""
    mock_model = get_mock_base_chat_model(response=MemoryExtractionModelOutput(inputs=[]))

    (_, plan_run) = get_test_plan_run()
    plan_run.outputs.step_outputs = {
        "$local_input": LocalOutput(value="Local value"),
        "$remote_input1": AgentMemoryOutput(
            output_name="remote_input1",
            plan_run_id=plan_run.id,
            summary="Summary 1",
        ),
    }

    agent = SimpleNamespace()
    agent.step = Step(
        task="Parse data from inputs",
        inputs=[
            Variable(name="local_input", description="Local input description"),
            Variable(name="remote_input1", description="Remote input description"),
        ],
        output="$out",
    )
    agent.tool = SimpleNamespace(
        id="TOOL_ID",
        name="TOOL_NAME",
        args_json_schema=_TestToolSchema.model_json_schema,
        args_schema=_TestToolSchema,
        description="TOOL_DESCRIPTION",
    )
    agent.plan_run = plan_run
    agent.get_system_context = mock.Mock(return_value="MOCKED_SYSTEM_CONTEXT")

    memory_model = MemoryExtractionModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,
    )

    with mock.patch(
        "portia.execution_agents.default_execution_agent.get_execution_context",
    ) as mock_context:
        memory_model.invoke({})

        assert mock_model.invoke.called
        assert mock_context.return_value.plan_run_context == "MOCKED_SYSTEM_CONTEXT"


def test_memory_extraction_model_with_retrieval() -> None:
    """Test memory extraction with some inputs requiring full value retrieval."""
    mock_model = get_mock_base_chat_model(
        response=MemoryExtractionModelOutput(inputs=["remote_input1", "remote_input2"]),
    )

    (_, plan_run) = get_test_plan_run()

    local_output = LocalOutput(value="Local value")
    remote_output1 = AgentMemoryOutput(
        output_name="$remote_input1",
        plan_run_id=plan_run.id,
        summary="Summary 1",
    )
    remote_output2 = AgentMemoryOutput(
        output_name="$remote_input2",
        plan_run_id=plan_run.id,
        summary="Summary 2",
    )
    remote_output3 = AgentMemoryOutput(
        output_name="$remote_input3",
        plan_run_id=plan_run.id,
        summary="Summary 3",
    )

    plan_run.outputs.step_outputs = {
        "$local_input": local_output,
        "$remote_input1": remote_output1,
        "$remote_input2": remote_output2,
        "$remote_input3": remote_output3,
    }

    agent = SimpleNamespace()
    agent.step = Step(
        task="Process data from inputs",
        inputs=[
            Variable(name="$local_input", description="Local input description"),
            Variable(name="$remote_input1", description="Remote input 1 description"),
            Variable(name="$remote_input2", description="Remote input 2 description"),
            Variable(name="$remote_input3", description="Remote input 3 description"),
        ],
        output="$out",
    )
    agent.tool = SimpleNamespace(
        id="TOOL_ID",
        name="TOOL_NAME",
        args_json_schema=_TestToolSchema.model_json_schema,
        args_schema=_TestToolSchema,
        description="TOOL_DESCRIPTION",
    )
    agent.plan_run = plan_run
    agent.get_system_context = mock.Mock(return_value="MOCKED_SYSTEM_CONTEXT")

    mock_memory = InMemoryStorage()
    retrieved_output1 = LocalOutput(value="Full remote value 1")
    retrieved_output2 = LocalOutput(value="Full remote value 2")
    agent.agent_memory = mock_memory

    def mock_get_plan_run_output(name: str, _: str) -> Output | None:
        if name == "$remote_input1":
            return retrieved_output1
        if name == "$remote_input2":
            return retrieved_output2
        return None

    memory_model = MemoryExtractionModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,
    )

    with (
        mock.patch.object(
            mock_memory,
            "get_plan_run_output",
            side_effect=mock_get_plan_run_output,
        ),
        mock.patch(
            "portia.execution_agents.default_execution_agent.get_execution_context",
        ) as mock_context,
    ):
        memory_model.invoke({})

        assert mock_model.invoke.called
        assert mock_context.return_value.plan_run_context == "MOCKED_SYSTEM_CONTEXT"
        assert mock_context.return_value.plan_run_context == "MOCKED_SYSTEM_CONTEXT"
        agent.get_system_context.assert_called_with(
            {
                "$local_input": local_output,
                "$remote_input1": retrieved_output1,
                "$remote_input2": retrieved_output2,
                "$remote_input3": remote_output3,
            },
        )


def test_memory_extraction_model_no_tool_error() -> None:
    """Test memory extraction model raises error when no tool is provided."""
    agent_no_tool = SimpleNamespace()
    agent_no_tool.tool = None

    memory_model_no_tool = MemoryExtractionModel(
        model=LangChainGenerativeModel(client=get_mock_base_chat_model(), model_name="test"),
        agent=agent_no_tool,
    )
    with pytest.raises(InvalidPlanRunStateError):
        memory_model_no_tool.invoke({})


def test_memory_extraction_model_validation_error() -> None:
    """Test memory extraction model handles model validation errors gracefully."""
    mock_model = get_mock_base_chat_model(response="INVALID_RESPONSE")

    (_, plan_run) = get_test_plan_run()
    plan_run.outputs.step_outputs = {}

    agent = SimpleNamespace()
    agent.step = Step(task="DESCRIPTION_STRING", output="$out")
    agent.tool = SimpleNamespace(
        id="TOOL_ID",
        name="TOOL_NAME",
        args_json_schema=_TestToolSchema.model_json_schema,
        args_schema=_TestToolSchema,
        description="TOOL_DESCRIPTION",
    )
    agent.plan_run = plan_run
    agent.get_system_context = mock.Mock(return_value="MOCKED_SYSTEM_CONTEXT")

    memory_model = MemoryExtractionModel(
        model=LangChainGenerativeModel(client=mock_model, model_name="test"),
        agent=agent,
    )

    with mock.patch(
        "portia.execution_agents.default_execution_agent.get_execution_context",
    ) as mock_context:
        memory_model.invoke({})
        assert mock_context.return_value.plan_run_context == "MOCKED_SYSTEM_CONTEXT"
