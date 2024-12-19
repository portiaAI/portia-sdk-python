"""Test simple agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from portia.agents.complex_langgraph_agent import (
    ToolCallingModel,
)
from portia.agents.simple_agent import SimpleAgent, SimpleToolCallingModel
from portia.agents.toolless_agent import ToolLessModel
from portia.config import Config
from portia.llm_wrapper import LLMWrapper
from portia.tool import Output
from tests.utils import AdditionTool

if TYPE_CHECKING:
    import pytest
    from langchain_core.prompt_values import ChatPromptValue
    from langchain_core.runnables.config import RunnableConfig


class MockInvoker:
    """Override for invoker."""

    called: bool = False
    prompt: ChatPromptValue | None = None

    class MockAIMessage:
        """Mock class for AIMessage."""

        def __init__(self, content: str) -> None:
            """Initialize content."""
            self.content = content

    def invoke(self, prompt: ChatPromptValue, _: RunnableConfig) -> MockAIMessage:
        """Mock run for invoking the chain."""
        self.called = True
        self.prompt = prompt
        return self.MockAIMessage("invoked")


def test_toolless_agent_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running an agent without a tool."""

    def toolless_model(self, state) -> dict[str, Any]:  # noqa: ANN001, ARG001
        response = AIMessage(
            content="This is a sentence that should never be hallucinated by the LLM.",
        )
        return {"messages": [response]}

    monkeypatch.setattr(ToolLessModel, "invoke", toolless_model)

    agent = SimpleAgent(
        description="Write a sentence with every letter of the alphabet.",
        inputs=[],
        tool=None,
        clarifications=[],
        system_context=[],
    )

    output = agent.execute_sync(llm=LLMWrapper(Config()).to_langchain(), step_outputs={})
    assert isinstance(output, Output)
    assert isinstance(output.value, str)
    assert output.value == "This is a sentence that should never be hallucinated by the LLM."


def test_basic_agent_task(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(SimpleToolCallingModel, "invoke", tool_calling_model)

    def tool_call(self, input, config) -> dict[str, Any]:  # noqa: A002, ANN001, ARG001
        return {
            "messages": ToolMessage(
                content="Sent email",
                artifact=Output(value="Sent email with id: 0"),
                tool_call_id="call_3z9rYHY6Rui7rTW0O7N7Wz51",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    agent = SimpleAgent(
        description="Send an email to test@example.com saying Hi as both the subject and body.",
        inputs=[],
        tool=AdditionTool(),
        clarifications=[],
        system_context=[],
    )

    output = agent.execute_sync(llm=LLMWrapper(Config()).to_langchain(), step_outputs={})
    assert isinstance(output, Output)
    assert output.value == "Sent email with id: 0"
