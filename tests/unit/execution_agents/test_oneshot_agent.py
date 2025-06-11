"""Test simple agent."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from portia.clarification import InputClarification
from portia.config import FEATURE_FLAG_ONE_SHOT_AGENT_CLARIFICATIONS_ENABLED
from portia.end_user import EndUser
from portia.errors import InvalidAgentError
from portia.execution_agents.context import StepInput
from portia.execution_agents.memory_extraction import MemoryExtractionStep
from portia.execution_agents.one_shot_agent import OneShotAgent, OneShotToolCallingModel
from portia.execution_agents.output import LocalDataValue, OutputDataValue
from portia.execution_hooks import ExecutionHooks
from portia.plan import ReadOnlyStep, Variable
from portia.plan_run import ReadOnlyPlanRun
from portia.prefixed_uuid import PlanRunUUID
from portia.storage import InMemoryStorage
from portia.tool import ToolRunContext
from tests.utils import (
    AdditionTool,
    get_mock_generative_model,
    get_test_config,
    get_test_plan_run,
)


def test_oneshot_agent_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running an agent without a tool.

    Note: This tests mocks almost everything, but allows us to make sure things
    are running in order and being called correctly and passed out correctly.
    """
    # Add mock for telemetry capture
    mock_telemetry = mock.MagicMock()

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

    def tool_call(self, input, config) -> dict[str, Any]:  # noqa: A002, ANN001, ARG001
        return {
            "messages": ToolMessage(
                content="Sent email",
                artifact=LocalDataValue(value="Sent email with id: 0"),
                tool_call_id="call_3z9rYHY6Rui7rTW0O7N7Wz51",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    (plan, plan_run) = get_test_plan_run()
    mock_before_tool_call = mock.MagicMock(return_value=None)
    mock_after_tool_call = mock.MagicMock(return_value=None)
    tool = AdditionTool()
    agent = OneShotAgent(
        plan=plan,
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=tool,
        execution_hooks=ExecutionHooks(
            before_tool_call=mock_before_tool_call,
            after_tool_call=mock_after_tool_call,
        ),
    )
    agent.telemetry = mock_telemetry

    output = agent.execute_sync()

    # Verify telemetry was captured with correct tool ID
    mock_telemetry.capture.assert_called_once()
    call_args = mock_telemetry.capture.call_args[0][0]
    assert call_args.tool_id == tool.id

    assert isinstance(output, OutputDataValue)
    assert output.get_value() == "Sent email with id: 0"
    mock_before_tool_call.assert_called_once_with(
        tool,
        {
            "recipients": ["test@example.com"],
            "email_title": "Hi",
            "email_body": "Hi",
        },
        ReadOnlyPlanRun.from_plan_run(agent.plan_run),
        ReadOnlyStep.from_step(agent.step),
    )
    mock_after_tool_call.assert_called_once_with(
        tool,
        "Sent email",
        ReadOnlyPlanRun.from_plan_run(agent.plan_run),
        ReadOnlyStep.from_step(agent.step),
    )


def test_oneshot_agent_without_tool_raises() -> None:
    """Test oneshot agent without tool raises."""
    (plan, plan_run) = get_test_plan_run()
    with pytest.raises(InvalidAgentError):
        OneShotAgent(
            plan=plan,
            plan_run=plan_run,
            end_user=EndUser(external_id="123"),
            config=get_test_config(),
            agent_memory=InMemoryStorage(),
            tool=None,
        ).execute_sync()


def test_oneshot_before_tool_call_with_clarification(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def tool_call(self, input, config) -> dict[str, Any]:  # noqa: A002, ANN001, ARG001
        nonlocal tool_node_called
        tool_node_called = True
        return {
            "messages": ToolMessage(
                content="3",
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
                user_guidance="Need clarification before tool call",
                step=plan_run.current_step_index,
                argument_name="num1",
                source="Test oneshot agent",
            )
        return None

    (plan, plan_run) = get_test_plan_run()

    # First execution - should return clarification
    agent = OneShotAgent(
        plan=plan,
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=AdditionTool(),
        execution_hooks=ExecutionHooks(
            before_tool_call=before_tool_call,
        ),
    )
    output = agent.execute_sync()

    assert tool_node_called is False
    assert len(output.get_value()) == 1  # pyright: ignore[reportArgumentType]
    output_value = output.get_value()[0]  # pyright: ignore[reportOptionalSubscript]
    assert isinstance(output_value, InputClarification)
    assert output_value.user_guidance == "Need clarification before tool call"

    # Second execution - should call the tool
    return_clarification = False
    tool_node_called = False
    agent.new_clarifications = []
    output = agent.execute_sync()

    assert tool_node_called is True
    assert output.get_value() == 3


def test_oneshot_after_tool_call_with_clarification(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def tool_call(self, input, config) -> dict[str, Any]:  # noqa: A002, ANN001, ARG001
        nonlocal tool_node_called
        tool_node_called = True
        return {
            "messages": ToolMessage(
                content="3",
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
                source="Test oneshot agent",
            )
        return None

    (plan, plan_run) = get_test_plan_run()

    # First execution - should return clarification after tool call
    agent = OneShotAgent(
        plan=plan,
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=AdditionTool(),
        execution_hooks=ExecutionHooks(
            after_tool_call=after_tool_call,
        ),
    )
    output = agent.execute_sync()

    assert tool_node_called is True
    assert len(output.get_value()) == 1  # pyright: ignore[reportArgumentType]
    output_value = output.get_value()[0]  # pyright: ignore[reportOptionalSubscript]
    assert isinstance(output_value, InputClarification)
    assert output_value.user_guidance == "Need clarification after tool call"

    # Second execution - should call the tool
    return_clarification = False
    tool_node_called = False
    agent.new_clarifications = []
    output = agent.execute_sync()

    assert tool_node_called is True
    assert output.get_value() == 3


def test_oneshot_agent_calls_clarification_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the oneshot agent correctly calls the clarification tool when needed."""
    model_response = AIMessage(content="")
    model_response.tool_calls = [
        {
            "name": "clarification_tool",
            "type": "tool_call",
            "id": "call_123",
            "args": {
                "argument_name": "missing_arg",
            },
        },
    ]
    mock_model = get_mock_generative_model(response=model_response)
    monkeypatch.setattr("portia.config.Config.get_execution_model", lambda self: mock_model)  # noqa: ARG005

    tool_node_called = False

    def tool_call(self, input, config) -> dict[str, Any]:  # noqa: A002, ANN001, ARG001
        nonlocal tool_node_called
        tool_node_called = True
        return {
            "messages": ToolMessage(
                content=InputClarification(
                    plan_run_id=PlanRunUUID(),
                    user_guidance="Missing Argument: missing_arg",
                    step=0,
                    argument_name="missing_arg",
                    source="Test oneshot agent",
                ).model_dump_json(),
                tool_call_id="call_123",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    (plan, plan_run) = get_test_plan_run()
    agent = OneShotAgent(
        plan=plan,
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(
            feature_flags={FEATURE_FLAG_ONE_SHOT_AGENT_CLARIFICATIONS_ENABLED: True}
        ),
        agent_memory=InMemoryStorage(),
        tool=AdditionTool(),
    )

    output = agent.execute_sync()

    assert tool_node_called is True
    assert len(output.get_value()) == 1  # pyright: ignore[reportArgumentType]
    output_value = output.get_value()[0]  # pyright: ignore[reportOptionalSubscript]
    assert isinstance(output_value, InputClarification)
    assert output_value.argument_name == "missing_arg"
    assert output_value.user_guidance == "Missing Argument: missing_arg"
    assert output_value.step == 0


def test_oneshot_agent_templates_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the oneshot agent correctly templates values before calling the tool."""
    model_response = AIMessage(content="")
    model_response.tool_calls = [
        {
            "name": "Send_Email_Tool",
            "type": "tool_call",
            "id": "call_123",
            "args": {
                "recipients": ["{{$email}}"],
                "email_title": "Hello {{$name}}",
                "email_body": "Dear {{$name}},\n\nThis is a test email.",
            },
        },
    ]
    mock_model = get_mock_generative_model(response=model_response)
    monkeypatch.setattr("portia.config.Config.get_execution_model", lambda self: mock_model)  # noqa: ARG005

    tool_node_called = False
    tool_args = None

    def tool_call(self, input, config) -> dict[str, Any]:  # noqa: A002, ANN001, ARG001
        nonlocal tool_node_called, tool_args
        tool_node_called = True
        tool_args = input["messages"][0].tool_calls[0]["args"]
        return {
            "messages": ToolMessage(
                content="Email sent",
                artifact=LocalDataValue(value="Email sent successfully"),
                tool_call_id="call_123",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    (plan, plan_run) = get_test_plan_run()
    step = plan.steps[0]
    step.inputs = [
        Variable(name="$email", description="User's email"),
        Variable(name="$name", description="User's name"),
    ]
    plan_run.plan_run_inputs = {
        "$email": LocalDataValue(value="test@example.com"),
        "$name": LocalDataValue(value="John Doe"),
    }

    agent = OneShotAgent(
        plan=plan,
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=AdditionTool(),
    )

    output = agent.execute_sync()

    assert tool_node_called is True
    assert tool_args is not None
    assert tool_args["recipients"] == ["test@example.com"]
    assert tool_args["email_title"] == "Hello John Doe"
    assert tool_args["email_body"] == "Dear John Doe,\n\nThis is a test email."
    assert output.get_value() == "Email sent successfully"


def test_oneshot_model_fails_without_tool() -> None:
    """Test that the oneshot model fails without a tool."""
    (plan, plan_run) = get_test_plan_run()
    agent = OneShotAgent(
        plan=plan,
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
    )
    tool_context = ToolRunContext(
        end_user=agent.end_user,
        plan=agent.plan,
        plan_run=agent.plan_run,
        config=agent.config,
        clarifications=agent.plan_run.get_clarifications_for_step(),
    )
    tool_calling_model = OneShotToolCallingModel(
        get_test_config().get_execution_model(), [], agent, tool_context
    )

    with pytest.raises(InvalidAgentError):
        tool_calling_model.invoke({"messages": [], "step_inputs": []})
