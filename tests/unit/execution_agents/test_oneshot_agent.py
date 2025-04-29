"""Test simple agent."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from portia.end_user import EndUser
from portia.errors import InvalidAgentError
from portia.execution_agents.context import StepInput
from portia.execution_agents.memory_extraction import MemoryExtractionStep
from portia.execution_agents.one_shot_agent import OneShotAgent, OneShotToolCallingModel
from portia.execution_agents.output import LocalOutput, Output
from portia.storage import InMemoryStorage
from tests.utils import AdditionTool, get_test_config, get_test_plan_run


def test_oneshot_agent_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running an agent without a tool.

    Note: This tests mocks almost everything, but allows us to make sure things
    are running in order and being called correctly and passed out correctly.
    """

    def memory_extraction_step(self, _) -> dict[str, Any]:  # noqa: ANN001, ARG001
        return {
            "step_inputs": [
                StepInput(
                    name="previous_input",
                    value="previous value",
                    description="Previous step input",
                )
            ]
        }

    monkeypatch.setattr(MemoryExtractionStep, "invoke", memory_extraction_step)

    def tool_calling_model(self, state) -> dict[str, Any]:  # noqa: ANN001, ARG001
        # Verify memory extraction step was called
        assert "step_inputs" in state
        assert len(state["step_inputs"]) == 1
        assert state["step_inputs"][0].name == "previous_input"
        assert state["step_inputs"][0].value == "previous value"

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
                artifact=LocalOutput(value="Sent email with id: 0"),
                tool_call_id="call_3z9rYHY6Rui7rTW0O7N7Wz51",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    (plan, plan_run) = get_test_plan_run()
    agent = OneShotAgent(
        step=plan.steps[0],
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=AdditionTool(),
    )

    output = agent.execute_sync()
    assert isinstance(output, Output)
    assert output.get_value() == "Sent email with id: 0"


def test_oneshot_agent_without_tool_raises() -> None:
    """Test oneshot agent without tool raises."""
    (plan, plan_run) = get_test_plan_run()
    with pytest.raises(InvalidAgentError):
        OneShotAgent(
            step=plan.steps[0],
            plan_run=plan_run,
            end_user=EndUser(external_id="123"),
            config=get_test_config(),
            agent_memory=InMemoryStorage(),
            tool=None,
        ).execute_sync()
