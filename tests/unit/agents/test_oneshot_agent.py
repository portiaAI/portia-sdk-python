"""Test simple agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from portia.agents.base_agent import Output
from portia.agents.one_shot_agent import OneShotAgent, OneShotToolCallingModel
from portia.agents.toolless_agent import ToolLessModel
from tests.utils import AdditionTool, get_test_config, get_test_workflow


def test_toolless_agent_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running an agent without a tool."""

    def toolless_model(self, state) -> dict[str, Any]:  # noqa: ANN001, ARG001
        response = AIMessage(
            content="This is a sentence that should never be hallucinated by the LLM.",
        )
        return {"messages": [response]}

    monkeypatch.setattr(ToolLessModel, "invoke", toolless_model)

    (plan, workflow) = get_test_workflow()
    agent = OneShotAgent(plan, step=plan.steps[0], workflow=workflow, config=get_test_config())

    output = agent.execute_sync()
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

    (plan, workflow) = get_test_workflow()
    agent = OneShotAgent(
        plan=plan,
        step=plan.steps[0],
        workflow=workflow,
        config=get_test_config(),
        tool=AdditionTool(),
    )

    output = agent.execute_sync()
    assert isinstance(output, Output)
    assert output.value == "Sent email with id: 0"
