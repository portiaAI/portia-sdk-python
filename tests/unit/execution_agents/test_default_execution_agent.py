"""test default execution agent."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from portia.clarification import InputClarification
from portia.errors import InvalidAgentError, InvalidPlanRunStateError
from portia.execution_agents.base_execution_agent import Output
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
from portia.plan import Step
from portia.tool import Tool
from tests.utils import (
    AdditionTool,
    get_test_config,
    get_test_llm_wrapper,
    get_test_plan_run,
    get_test_tool_context,
)

if TYPE_CHECKING:
    from langchain_core.prompt_values import ChatPromptValue
    from langchain_core.runnables.config import RunnableConfig


@pytest.fixture(scope="session", autouse=True)
def _setup() -> None:
    logging.basicConfig(level=logging.INFO)


class MockInvoker:
    """Mock invoker."""

    called: bool
    prompt: ChatPromptValue | None
    response: AIMessage | BaseModel | None
    output_format: Any | None
    tools: Any | None
    method: str | None

    def __init__(self, response: AIMessage | BaseModel | None = None) -> None:
        """Init worker."""
        self.called = False
        self.prompt = None
        self.response = response
        self.output_format = None
        self.tools = None
        self.method = None

    def invoke(
        self,
        prompt: ChatPromptValue,
        _: RunnableConfig | None = None,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> AIMessage | BaseModel:
        """Mock run for invoking the chain."""
        self.called = True
        self.prompt = prompt
        if self.response:
            return self.response
        return AIMessage(content="invoked")

    def with_structured_output(
        self,
        output_format: Any,  # noqa: ANN401
        method: str = "function_calling",
    ) -> MockInvoker:
        """Model wrapper for structured output."""
        self.output_format = output_format
        self.method = method
        return self


class _TestToolSchema(BaseModel):
    """Input for TestTool."""

    content: str = Field(..., description="INPUT_DESCRIPTION")


def test_parser_model(monkeypatch: pytest.MonkeyPatch) -> None:
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
    mock_invoker = MockInvoker(response=tool_inputs)
    monkeypatch.setattr(ChatOpenAI, "invoke", mock_invoker.invoke)
    monkeypatch.setattr(ChatOpenAI, "with_structured_output", mock_invoker.with_structured_output)

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
        llm=get_test_llm_wrapper().to_langchain(),
        context="CONTEXT_STRING",
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    parser_model.invoke({})  # type: ignore  # noqa: PGH003

    assert mock_invoker.called
    messages = mock_invoker.prompt
    assert messages
    assert "You are a highly capable assistant" in messages[0].content  # type: ignore  # noqa: PGH003
    assert "CONTEXT_STRING" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "DESCRIPTION_STRING" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_NAME" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_DESCRIPTION" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "INPUT_DESCRIPTION" in messages[1].content  # type: ignore  # noqa: PGH003
    assert mock_invoker.output_format == ToolInputs


def test_parser_model_with_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the parser model with retries."""
    tool_inputs = ToolInputs(
        args=[],
    )
    mock_invoker = MockInvoker(response=tool_inputs)
    monkeypatch.setattr(ChatOpenAI, "invoke", mock_invoker.invoke)
    monkeypatch.setattr(ChatOpenAI, "with_structured_output", mock_invoker.with_structured_output)

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
        llm=get_test_llm_wrapper().to_langchain(),
        context="CONTEXT_STRING",
        agent=agent,  # type: ignore  # noqa: PGH003
    )

    # Track number of invoke calls
    invoke_count = 0
    original_invoke = parser_model.invoke

    def counted_invoke(*args, **kwargs) -> dict[str, Any]:  # noqa: ANN002, ANN003
        nonlocal invoke_count
        invoke_count += 1
        return original_invoke(*args, **kwargs)

    parser_model.invoke = counted_invoke

    parser_model.invoke({})  # type: ignore  # noqa: PGH003

    # Initial invoke call is not counted towards the MAX_RETRIES
    assert invoke_count == MAX_RETRIES + 1


def test_parser_model_with_retries_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the parser model handling of invalid JSON and retries."""
    mock_invoker = MockInvoker(response=AIMessage(content="INVALID_JSON"))
    monkeypatch.setattr(ChatOpenAI, "invoke", mock_invoker.invoke)
    monkeypatch.setattr(ChatOpenAI, "with_structured_output", mock_invoker.with_structured_output)

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
        llm=get_test_llm_wrapper().to_langchain(),
        context="CONTEXT_STRING",
        agent=agent,  # type: ignore  # noqa: PGH003
    )

    # Track number of invoke calls
    invoke_count = 0
    original_invoke = parser_model.invoke

    def counted_invoke(*args, **kwargs) -> dict[str, Any]:  # noqa: ANN002, ANN003
        nonlocal invoke_count
        invoke_count += 1
        return original_invoke(*args, **kwargs)

    parser_model.invoke = counted_invoke

    parser_model.invoke({})  # type: ignore  # noqa: PGH003

    # Initial invoke call is not counted towards the MAX_RETRIES
    assert invoke_count == MAX_RETRIES + 1


def test_parser_model_with_invalid_args(monkeypatch: pytest.MonkeyPatch) -> None:
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

    class MockInvoker:
        def invoke(self, *_, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG002
            nonlocal current_response_index
            response = responses[current_response_index]
            current_response_index += 1
            return response

        def with_structured_output(
            self,
            output_format: Any,
            method: str = "function_calling",  # noqa: ANN401 ARG002
        ) -> MockInvoker:
            return self

    monkeypatch.setattr(ChatOpenAI, "invoke", MockInvoker().invoke)
    monkeypatch.setattr(ChatOpenAI, "with_structured_output", MockInvoker().with_structured_output)

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
        llm=get_test_llm_wrapper().to_langchain(),
        context="CONTEXT_STRING",
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


def test_verifier_model(monkeypatch: pytest.MonkeyPatch) -> None:
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
    mockinvoker = MockInvoker(response=verified_tool_inputs)
    monkeypatch.setattr(ChatOpenAI, "invoke", mockinvoker.invoke)
    monkeypatch.setattr(ChatOpenAI, "with_structured_output", mockinvoker.with_structured_output)

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
        llm=get_test_llm_wrapper().to_langchain(),
        context="CONTEXT_STRING",
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    verifier_model.invoke({"messages": [AIMessage(content=tool_inputs.model_dump_json(indent=2))]})

    assert mockinvoker.called
    messages = mockinvoker.prompt
    assert "You are an expert reviewer" in messages[0].content  # type: ignore  # noqa: PGH003
    assert "CONTEXT_STRING" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "DESCRIPTION_STRING" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_NAME" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "INPUT_DESCRIPTION" in messages[1].content  # type: ignore  # noqa: PGH003
    assert mockinvoker.output_format == VerifiedToolInputs


def test_verifier_model_schema_validation(monkeypatch: pytest.MonkeyPatch) -> None:
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
    mockinvoker = MockInvoker(response=verified_tool_inputs)
    monkeypatch.setattr(ChatOpenAI, "invoke", mockinvoker.invoke)
    monkeypatch.setattr(ChatOpenAI, "with_structured_output", mockinvoker.with_structured_output)

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
        llm=get_test_llm_wrapper().to_langchain(),
        context="CONTEXT_STRING",
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


def test_tool_calling_model_no_hallucinations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the tool calling model."""
    verified_tool_inputs = VerifiedToolInputs(
        args=[VerifiedToolArgument(name="content", value="CONTENT_STRING", made_up=False)],
    )
    mockinvoker = MockInvoker(
        response=SimpleNamespace(tool_calls=[{"name": "add_tool", "args": "CALL_ARGS"}]),  # type: ignore  # noqa: PGH003
    )
    monkeypatch.setattr(ChatOpenAI, "invoke", mockinvoker.invoke)
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
        llm=get_test_llm_wrapper().to_langchain(),
        context="CONTEXT_STRING",
        tools=[AdditionTool().to_langchain_with_artifact(ctx=get_test_tool_context())],
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    tool_calling_model.invoke({"messages": []})

    assert mockinvoker.called
    messages = mockinvoker.prompt
    assert "You are very powerful assistant" in messages[0].content  # type: ignore  # noqa: PGH003
    assert "CONTEXT_STRING" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "DESCRIPTION_STRING" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_NAME" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "INPUT_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003


def test_tool_calling_model_with_hallucinations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the tool calling model."""
    verified_tool_inputs = VerifiedToolInputs(
        args=[VerifiedToolArgument(name="content", value="CONTENT_STRING", made_up=True)],
    )
    mockinvoker = MockInvoker(
        response=SimpleNamespace(tool_calls=[{"name": "add_tool", "args": "CALL_ARGS"}]),  # type: ignore  # noqa: PGH003
    )
    monkeypatch.setattr(ChatOpenAI, "invoke", mockinvoker.invoke)

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
        llm=get_test_llm_wrapper().to_langchain(),
        context="CONTEXT_STRING",
        tools=[AdditionTool().to_langchain_with_artifact(ctx=get_test_tool_context())],
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    tool_calling_model.invoke({"messages": []})

    assert mockinvoker.called
    messages = mockinvoker.prompt
    assert messages
    assert "You are very powerful assistant" in messages[0].content  # type: ignore  # noqa: PGH003
    assert "CONTEXT_STRING" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "DESCRIPTION_STRING" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_NAME" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "INPUT_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003


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
                artifact=Output(value="Sent email with id: 0"),
                tool_call_id="call_3z9rYHY6Rui7rTW0O7N7Wz51",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    (plan, plan_run) = get_test_plan_run()
    agent = DefaultExecutionAgent(
        step=plan.steps[0],
        plan_run=plan_run,
        config=get_test_config(),
        tool=tool,
    )

    output = agent.execute_sync()
    assert isinstance(output, Output)
    assert output.value == "Sent email with id: 0"


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
                artifact=Output(value="Sent email with id: 0"),
                tool_call_id="call_3z9rYHY6Rui7rTW0O7N7Wz51",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    (plan, plan_run) = get_test_plan_run()
    agent = DefaultExecutionAgent(
        step=plan.steps[0],
        plan_run=plan_run,
        config=get_test_config(),
        tool=tool,
    )
    agent.verified_args = verified_tool_inputs

    output = agent.execute_sync()
    assert isinstance(output, Output)
    assert output.value == "Sent email with id: 0"


def test_default_execution_agent_edge_cases() -> None:
    """Tests edge cases are handled."""
    agent = SimpleNamespace()
    agent.step = Step(task="DESCRIPTION_STRING", output="$out")
    agent.tool = None
    parser_model = ParserModel(
        llm=get_test_llm_wrapper().to_langchain(),
        context="CONTEXT_STRING",
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    with pytest.raises(InvalidPlanRunStateError):
        parser_model.invoke({"messages": []})

    agent.verified_args = None
    tool_calling_model = ToolCallingModel(
        llm=get_test_llm_wrapper().to_langchain(),
        context="CONTEXT_STRING",
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
        tool=MockTool(),
    )
    model = VerifierModel(
        llm=get_test_llm_wrapper().to_langchain(),
        context="CONTEXT_STRING",
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
        llm=get_test_llm_wrapper().to_langchain(),
        context="CONTEXT_STRING",
        agent=agent,  # type: ignore  # noqa: PGH003
    )

    # Check error with no tool specified
    agent.tool = None
    with pytest.raises(InvalidPlanRunStateError):
        verifier_model.invoke({"messages": []})
