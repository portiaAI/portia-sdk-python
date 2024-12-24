"""Test simple agent."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from portia.agents.one_shot_agent import OneShotAgent, OneShotToolCallingModel
from portia.agents.toolless_agent import ToolLessModel
from portia.clarification import InputClarification
from portia.config import default_config
from portia.errors import InvalidAgentOutputError
from portia.llm_wrapper import LLMWrapper
from portia.tool import Output
from tests.utils import AdditionTool


def test_toolless_agent_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running an agent without a tool."""

    def toolless_model(self, state) -> dict[str, Any]:  # noqa: ANN001, ARG001
        response = AIMessage(
            content="This is a sentence that should never be hallucinated by the LLM.",
        )
        return {"messages": [response]}

    monkeypatch.setattr(ToolLessModel, "invoke", toolless_model)

    agent = OneShotAgent(
        description="Write a sentence with every letter of the alphabet.",
        inputs=[],
        tool=None,
        clarifications=[],
        system_context=[],
    )

    output = agent.execute_sync(llm=LLMWrapper(default_config()).to_langchain(), step_outputs={})
    assert isinstance(output, Output)
    assert isinstance(output.value, str)
    assert output.value == "This is a sentence that should never be hallucinated by the LLM."


def test_oneshot_agent_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running an agent without a tool.

    Note: This tests mocks almost everything, but allows us to make sure things
    are running in order and being called correctly and passed out correctly.
    """

    def tool_calling_model(self, state) -> dict[str, Any]:  # noqa: ANN001, ARG001
        response = AIMessage(content="")
        response.tool_calls = [
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
        return {"messages": [response]}

    monkeypatch.setattr(OneShotToolCallingModel, "invoke", tool_calling_model)

    def tool_call(self, input, config) -> dict[str, Any]:  # noqa: A002, ANN001, ARG001
        return {
            "messages": ToolMessage(
                content="Sent email",
                artifact=Output(value="Sent email with id: 0"),
                tool_call_id="call_3z9rYHY6Rui7rTW0O7N7Wz51",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    agent = OneShotAgent(
        description="Send an email to test@example.com saying Hi as both the subject and body.",
        inputs=[],
        tool=AdditionTool(),
        clarifications=[],
        system_context=[],
    )

    output = agent.execute_sync(llm=LLMWrapper(default_config()).to_langchain(), step_outputs={})
    assert isinstance(output, Output)
    assert output.value == "Sent email with id: 0"


def test_oneshot_agent_end_criteria() -> None:
    """Test process_output."""
    # check end criteria
    output = OneShotAgent.call_tool_or_return(
        {
            "messages": [
                HumanMessage(
                    content="Sent email",
                ),
            ],
        },
    )

    assert output == END


def test_oneshot_agent_process_output_clarification() -> None:
    """Test process_output."""
    # check process output when clarifications
    agent = OneShotAgent(
        description="Send an email to test@example.com saying Hi as both the subject and body.",
        inputs=[],
        tool=AdditionTool(),
        clarifications=[],
        system_context=[],
    )
    agent.new_clarifications = [InputClarification(user_guidance="test", argument_name="test")]
    output = agent.process_output(
        HumanMessage(
            content="Sent email",
        ),
    )
    assert isinstance(output, Output)
    assert isinstance(output.value, list)
    assert isinstance(output.value[0], InputClarification)


def test_oneshot_agent_process_output_tools() -> None:
    """Test process_output."""
    # check process output when clarifications
    agent = OneShotAgent(
        description="Send an email to test@example.com saying Hi as both the subject and body.",
        inputs=[],
        tool=AdditionTool(),
        clarifications=[],
        system_context=[],
    )
    message = ToolMessage(content="", tool_call_id="call_J")
    message.artifact = "123"
    output = agent.process_output(
        message,
    )
    assert isinstance(output, Output)
    assert isinstance(output.value, str)
    assert output.value == "123"

    message = ToolMessage(content="456", tool_call_id="call_J")
    output = agent.process_output(
        message,
    )
    assert isinstance(output, Output)
    assert isinstance(output.value, str)
    assert output.value == "456"

    message = HumanMessage(content="789")
    output = agent.process_output(
        message,
    )
    assert isinstance(output, Output)
    assert isinstance(output.value, str)
    assert output.value == "789"

    message = AIMessage(content="456")
    with pytest.raises(InvalidAgentOutputError):
        output = agent.process_output(
            message,
        )
