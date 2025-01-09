"""test verifier agent."""

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

from portia.agents.toolless_agent import ToolLessModel
from portia.agents.verifier_agent import (
    ParserModel,
    ToolArgument,
    ToolCallingModel,
    ToolInputs,
    VerifiedToolArgument,
    VerifiedToolInputs,
    VerifierAgent,
    VerifierModel,
)
from portia.clarification import InputClarification
from portia.config import default_config
from portia.errors import InvalidAgentOutputError, InvalidWorkflowStateError
from portia.llm_wrapper import LLMWrapper
from portia.tool import Output
from tests.utils import AdditionTool

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

    def __init__(self, response: AIMessage | BaseModel | None = None) -> None:
        """Init worker."""
        self.called = False
        self.prompt = None
        self.response = response
        self.output_format = None
        self.tools = None

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

    def with_structured_output(self, output_format: Any) -> MockInvoker:  # noqa: ANN401
        """Model wrapper for structured output."""
        self.output_format = output_format
        return self


def test_toolless_agent_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running an agent without a tool."""

    def toolless_model(self, state):  # noqa: ANN001, ANN202, ARG001
        response = AIMessage(
            content="This is a sentence that should never be hallucinated by the LLM.",
        )
        return {"messages": [response]}

    monkeypatch.setattr(ToolLessModel, "invoke", toolless_model)

    agent = VerifierAgent(
        description="Write a sentence with every letter of the alphabet.",
        inputs=[],
        tool=None,
        clarifications=[],
        system_context_extension=[],
    )

    output = agent.execute_sync(llm=LLMWrapper(default_config()).to_langchain(), step_outputs={})
    assert isinstance(output, Output)
    assert isinstance(output.value, str)
    assert output.value == "This is a sentence that should never be hallucinated by the LLM."


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

    agent = SimpleNamespace(description="DESCRIPTION_STRING")
    agent.tool = SimpleNamespace(
        name="TOOL_NAME",
        args_json_schema=_TestToolSchema.model_json_schema,
        description="TOOL_DESCRIPTION",
    )
    parser_model = ParserModel(
        llm=LLMWrapper(default_config()).to_langchain(),
        context="CONTEXT_STRING",
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    parser_model.invoke({})  # type: ignore  # noqa: PGH003

    assert mock_invoker.called
    messages = mock_invoker.prompt
    assert messages
    assert "You are very powerful assistant" in messages[0].content  # type: ignore  # noqa: PGH003
    assert "CONTEXT_STRING" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "DESCRIPTION_STRING" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_NAME" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_DESCRIPTION" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "INPUT_DESCRIPTION" in messages[1].content  # type: ignore  # noqa: PGH003
    assert mock_invoker.output_format == ToolInputs


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

    agent = SimpleNamespace(description="DESCRIPTION_STRING")
    agent.tool = SimpleNamespace(
        name="TOOL_NAME",
        args_json_schema=_TestToolSchema,
        description="TOOL_DESCRIPTION",
    )
    verifier_model = VerifierModel(
        llm=LLMWrapper(default_config()).to_langchain(),
        context="CONTEXT_STRING",
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    verifier_model.invoke({"messages": [AIMessage(content=tool_inputs.model_dump_json(indent=2))]})

    assert mockinvoker.called
    messages = mockinvoker.prompt
    assert "You are an expert reviewer" in messages[0].content  # type: ignore  # noqa: PGH003
    assert "CONTEXT_STRING" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "DESCRIPTION_STRING" in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_NAME" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "TOOL_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert "INPUT_DESCRIPTION" not in messages[1].content  # type: ignore  # noqa: PGH003
    assert mockinvoker.output_format == VerifiedToolInputs


def test_tool_calling_model_no_hallucinations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the tool calling model."""
    verified_tool_inputs = VerifiedToolInputs(
        args=[VerifiedToolArgument(name="content", value="CONTENT_STRING", made_up=False)],
    )
    mockinvoker = MockInvoker(
        response=SimpleNamespace(tool_calls=[{"name": "add_tool", "args": "CALL_ARGS"}]),  # type: ignore  # noqa: PGH003
    )
    monkeypatch.setattr(ChatOpenAI, "invoke", mockinvoker.invoke)

    agent = SimpleNamespace(
        description="DESCRIPTION_STRING",
        verified_args=verified_tool_inputs,
        clarifications=[],
    )
    agent.tool = SimpleNamespace(
        name="TOOL_NAME",
        args_json_schema=_TestToolSchema,
        description="TOOL_DESCRIPTION",
    )
    tool_calling_model = ToolCallingModel(
        llm=LLMWrapper(default_config()).to_langchain(),
        context="CONTEXT_STRING",
        tools=[AdditionTool().to_langchain(return_artifact=True)],
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

    clarification = InputClarification(
        user_guidance="USER_GUIDANCE",
        response="CLARIFICATION_RESPONSE",
        argument_name="content",
        resolved=True,
    )

    failed_clarification = InputClarification(
        user_guidance="USER_GUIDANCE_FAILED",
        response="FAILED",
        argument_name="content",
        resolved=True,
    )

    agent = SimpleNamespace(
        description="DESCRIPTION_STRING",
        verified_args=verified_tool_inputs,
        clarifications=[failed_clarification, clarification],
        missing_args={"content": clarification},
        get_last_resolved_clarification=lambda arg_name, arg_value: clarification
        if arg_name == "content" and arg_value == "CONTENT_STRING"
        else None,
    )
    agent.tool = SimpleNamespace(
        name="TOOL_NAME",
        args_json_schema=_TestToolSchema,
        description="TOOL_DESCRIPTION",
    )
    tool_calling_model = ToolCallingModel(
        llm=LLMWrapper(default_config()).to_langchain(),
        context="CONTEXT_STRING",
        tools=[AdditionTool().to_langchain(return_artifact=True)],
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    tool_calling_model.invoke({"messages": []})

    assert mockinvoker.called
    messages = mockinvoker.prompt
    assert messages
    assert "You are very powerful assistant" in messages[0].content  # type: ignore  # noqa: PGH003
    assert "CLARIFICATION_RESPONSE" in messages[1].content  # type: ignore  # noqa: PGH003
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

    agent = VerifierAgent(
        description="Send an email to test@example.com saying Hi as both the subject and body.",
        inputs=[],
        tool=tool,
        clarifications=[],
        system_context_extension=[],
    )

    output = agent.execute_sync(llm=LLMWrapper(default_config()).to_langchain(), step_outputs={})
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

    agent = VerifierAgent(
        description="Send an email to test@example.com saying Hi as both the subject and body.",
        inputs=[],
        tool=tool,
        clarifications=[],
        system_context_extension=[],
    )
    agent.verified_args = verified_tool_inputs

    output = agent.execute_sync(llm=LLMWrapper(default_config()).to_langchain(), step_outputs={})
    assert isinstance(output, Output)
    assert output.value == "Sent email with id: 0"


def test_verifier_agent_edge_cases() -> None:
    """Tests edge cases are handled."""
    agent = SimpleNamespace(description="DESCRIPTION_STRING")
    agent.tool = None
    parser_model = ParserModel(
        llm=LLMWrapper(default_config()).to_langchain(),
        context="CONTEXT_STRING",
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    with pytest.raises(InvalidWorkflowStateError):
        parser_model.invoke({"messages": []})

    agent.verified_args = None
    tool_calling_model = ToolCallingModel(
        llm=LLMWrapper(default_config()).to_langchain(),
        context="CONTEXT_STRING",
        tools=[AdditionTool().to_langchain(return_artifact=True)],
        agent=agent,  # type: ignore  # noqa: PGH003
    )
    with pytest.raises(InvalidWorkflowStateError):
        tool_calling_model.invoke({"messages": []})


def test_get_last_resolved_clarification() -> None:
    """Test get_last_resolved_clarification."""
    resolved_clarification1 = InputClarification(
        argument_name="arg",
        response="2",
        user_guidance="FAILED",
        resolved=True,
    )
    resolved_clarification2 = InputClarification(
        argument_name="arg",
        response="2",
        user_guidance="SUCCESS",
        resolved=True,
    )
    unresolved_clarification = InputClarification(
        argument_name="arg",
        response="2",
        user_guidance="",
        resolved=False,
    )
    agent = VerifierAgent(
        description="Send an email to test@example.com saying Hi as both the subject and body.",
        inputs=[],
        tool=None,
        clarifications=[resolved_clarification1, resolved_clarification2, unresolved_clarification],
        system_context_extension=[],
    )
    assert agent.get_last_resolved_clarification("arg", "2") == resolved_clarification2


def test_clarifications_or_continue() -> None:
    """Test clarifications_or_continue."""
    clarification = InputClarification(
        argument_name="arg",
        response="2",
        user_guidance="",
        resolved=True,
    )

    agent = VerifierAgent(
        description="Send an email to test@example.com saying Hi as both the subject and body.",
        inputs=[],
        tool=None,
        clarifications=[clarification],
        system_context_extension=[],
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
        argument_name="arg",
        response="1",
        user_guidance="",
        resolved=True,
    )

    agent = VerifierAgent(
        description="Send an email to test@example.com saying Hi as both the subject and body.",
        inputs=[],
        tool=None,
        clarifications=[clarification],
        system_context_extension=[],
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


def test_call_tool_or_return() -> None:
    """Test call_tool_or_return."""
    # when clarifications don't match expect a new one
    output = VerifierAgent.call_tool_or_return(
        {
            "messages": [
                HumanMessage(
                    content="",
                ),
            ],
        },
    )
    assert output == END


def test_process_output() -> None:
    """Test process_output."""
    agent = VerifierAgent(
        description="Send an email to test@example.com saying Hi as both the subject and body.",
        inputs=[],
        tool=None,
        clarifications=[],
        system_context_extension=[],
    )

    # with new clarifications
    clarification = InputClarification(
        argument_name="arg",
        response="1",
        user_guidance="",
        resolved=True,
    )
    agent.new_clarifications = [clarification]
    output = agent.process_output(
        HumanMessage(content=""),
    )
    assert isinstance(output, Output)
    assert isinstance(output.value, list)
    assert isinstance(output.value[0], InputClarification)

    # with no new clarifications
    agent.new_clarifications = []
    message = ToolMessage(content="", tool_call_id="123")
    message.artifact = "123"
    output = agent.process_output(message)
    assert isinstance(output, Output)
    assert output.value == "123"

    # with no artifact
    message = ToolMessage(content="456", tool_call_id="123")
    output = agent.process_output(message)
    assert isinstance(output, Output)
    assert output.value == "456"

    message = AIMessage(content="456")
    with pytest.raises(InvalidAgentOutputError):
        output = agent.process_output(
            message,
        )
