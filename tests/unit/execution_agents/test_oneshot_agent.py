"""Test simple agent."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from portia.clarification import InputClarification
from portia.end_user import EndUser
from portia.errors import InvalidAgentError
from portia.execution_agents.context import StepInput
from portia.execution_agents.memory_extraction import MemoryExtractionStep
from portia.execution_agents.one_shot_agent import OneShotAgent, OneShotToolCallingModel
from portia.execution_agents.output import LocalDataValue, Output
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

    assert mock_telemetry.capture.call_count == 2
    # Verify telemetry was captured with correct tool ID
    call_args = mock_telemetry.capture.call_args[0][0]
    assert call_args.tool_id == tool.id

    assert isinstance(output, Output)
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


@pytest.mark.asyncio
async def test_oneshot_agent_task_async(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running an agent without a tool (async version).

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

    async def tool_call(self, input, config) -> dict[str, Any]:  # noqa: A002, ANN001, ARG001
        return {
            "messages": ToolMessage(
                content="Sent email",
                artifact=LocalDataValue(value="Sent email with id: 0"),
                tool_call_id="call_3z9rYHY6Rui7rTW0O7N7Wz51",
            ),
        }

    monkeypatch.setattr(ToolNode, "ainvoke", tool_call)

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

    output = await agent.execute_async()

    assert mock_telemetry.capture.call_count == 2
    # Verify telemetry was captured with correct tool ID
    call_args = mock_telemetry.capture.call_args[0][0]
    assert call_args.tool_id == tool.id

    assert isinstance(output, Output)
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


@pytest.mark.asyncio
async def test_oneshot_agent_without_tool_raises_async() -> None:
    """Test oneshot agent without tool raises (async version)."""
    (plan, plan_run) = get_test_plan_run()
    with pytest.raises(InvalidAgentError):
        await OneShotAgent(
            plan=plan,
            plan_run=plan_run,
            end_user=EndUser(external_id="123"),
            config=get_test_config(),
            agent_memory=InMemoryStorage(),
            tool=None,
        ).execute_async()


def test_oneshot_before_tool_call_with_clarifications(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that before_tool_call can interrupt execution by returning multiple clarifications."""
    model_response = AIMessage(content="")
    model_response.tool_calls = [
        {
            "name": "Send_Email_Tool",
            "type": "tool_call",
            "id": "call_1",
            "args": {
                "recipients": ["test1@example.com"],
                "email_title": "Hi",
                "email_body": "Hi",
            },
        },
        {
            "name": "Send_Email_Tool",
            "type": "tool_call",
            "id": "call_2",
            "args": {
                "recipients": ["test2@example.com"],
                "email_title": "Hi",
                "email_body": "Hi",
            },
        },
        {
            "name": "Send_Email_Tool",
            "type": "tool_call",
            "id": "call_3",
            "args": {
                "recipients": ["test3@example.com"],
                "email_title": "Hi",
                "email_body": "Hi",
            },
        },
    ]
    mock_model = get_mock_generative_model(response=model_response)
    monkeypatch.setattr(
        "portia.config.Config.get_execution_model",
        lambda self: mock_model,  # noqa: ARG005
    )

    tool_node_called = False

    def tool_call(self, input, config) -> dict[str, Any]:  # noqa: A002, ANN001, ARG001
        nonlocal tool_node_called
        tool_node_called = True
        return {
            "messages": ToolMessage(
                content="3",
                artifact=LocalDataValue(value=3),
                tool_call_id="call_1",
            ),
        }

    monkeypatch.setattr(ToolNode, "invoke", tool_call)

    call_count = 0
    return_clarification = True

    def before_tool_call(tool, args, plan_run, step) -> InputClarification | None:  # noqa: ANN001, ARG001
        nonlocal call_count, return_clarification
        if not return_clarification:
            call_count += 1
            return None

        call_count += 1
        if call_count == 1:
            return InputClarification(
                plan_run_id=plan_run.id,
                user_guidance="Need clarification for num1",
                step=plan_run.current_step_index,
                argument_name="num1",
                source="Test oneshot agent",
            )
        if call_count == 2:
            return InputClarification(
                plan_run_id=plan_run.id,
                user_guidance="Need clarification for num2",
                step=plan_run.current_step_index,
                argument_name="num2",
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
    assert call_count == 3
    output_values = output.get_value()
    assert isinstance(output_values, list)
    assert len(output_values) == 2
    output_value_1 = output_values[0]
    assert isinstance(output_value_1, InputClarification)
    assert output_value_1.user_guidance == "Need clarification for num1"
    assert output_value_1.argument_name == "num1"
    output_value_2 = output_values[1]
    assert isinstance(output_value_2, InputClarification)
    assert output_value_2.user_guidance == "Need clarification for num2"
    assert output_value_2.argument_name == "num2"

    # Second execution - should call the tool
    return_clarification = False
    tool_node_called = False
    call_count = 0
    agent.new_clarifications = []
    output = agent.execute_sync()

    assert tool_node_called is True
    assert call_count == 3
    assert output.get_value() == 3


@pytest.mark.asyncio
async def test_oneshot_before_tool_call_with_clarifications_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that before_tool_call can interrupt by returning multiple clarifications async."""
    model_response = AIMessage(content="")
    model_response.tool_calls = [
        {
            "name": "Send_Email_Tool",
            "type": "tool_call",
            "id": "call_1",
            "args": {
                "recipients": ["test1@example.com"],
                "email_title": "Hi",
                "email_body": "Hi",
            },
        },
        {
            "name": "Send_Email_Tool",
            "type": "tool_call",
            "id": "call_2",
            "args": {
                "recipients": ["test2@example.com"],
                "email_title": "Hi",
                "email_body": "Hi",
            },
        },
        {
            "name": "Send_Email_Tool",
            "type": "tool_call",
            "id": "call_3",
            "args": {
                "recipients": ["test3@example.com"],
                "email_title": "Hi",
                "email_body": "Hi",
            },
        },
    ]
    mock_model = get_mock_generative_model(response=model_response)
    monkeypatch.setattr(
        "portia.config.Config.get_execution_model",
        lambda self: mock_model,  # noqa: ARG005
    )

    tool_node_called = False

    async def tool_call(self, input, config) -> dict[str, Any]:  # noqa: A002, ANN001, ARG001
        nonlocal tool_node_called
        tool_node_called = True
        return {
            "messages": ToolMessage(
                content="3",
                artifact=LocalDataValue(value=3),
                tool_call_id="call_1",
            ),
        }

    monkeypatch.setattr(ToolNode, "ainvoke", tool_call)

    call_count = 0
    return_clarification = True

    def before_tool_call(tool, args, plan_run, step) -> InputClarification | None:  # noqa: ANN001, ARG001
        nonlocal call_count, return_clarification
        if not return_clarification:
            call_count += 1
            return None

        call_count += 1
        if call_count == 1:
            return InputClarification(
                plan_run_id=plan_run.id,
                user_guidance="Need clarification for num1",
                step=plan_run.current_step_index,
                argument_name="num1",
                source="Test oneshot agent",
            )
        if call_count == 2:
            return InputClarification(
                plan_run_id=plan_run.id,
                user_guidance="Need clarification for num2",
                step=plan_run.current_step_index,
                argument_name="num2",
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
    output = await agent.execute_async()

    assert tool_node_called is False
    assert call_count == 3
    output_values = output.get_value()
    assert isinstance(output_values, list)
    assert len(output_values) == 2
    output_value_1 = output_values[0]
    assert isinstance(output_value_1, InputClarification)
    assert output_value_1.user_guidance == "Need clarification for num1"
    assert output_value_1.argument_name == "num1"
    output_value_2 = output_values[1]
    assert isinstance(output_value_2, InputClarification)
    assert output_value_2.user_guidance == "Need clarification for num2"
    assert output_value_2.argument_name == "num2"

    # Second execution - should call the tool
    return_clarification = False
    tool_node_called = False
    call_count = 0
    agent.new_clarifications = []
    output = await agent.execute_async()

    assert tool_node_called is True
    assert call_count == 3
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


@pytest.mark.asyncio
async def test_oneshot_after_tool_call_with_clarification_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that after_tool_call can interrupt execution by returning a clarification async."""
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

    async def tool_call(self, input, config) -> dict[str, Any]:  # noqa: A002, ANN001, ARG001
        nonlocal tool_node_called
        tool_node_called = True
        return {
            "messages": ToolMessage(
                content="3",
                artifact=LocalDataValue(value=3),
                tool_call_id="call_3z9rYHY6Rui7rTW0O7N7Wz51",
            ),
        }

    monkeypatch.setattr(ToolNode, "ainvoke", tool_call)

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
    output = await agent.execute_async()

    assert tool_node_called is True
    assert len(output.get_value()) == 1  # pyright: ignore[reportArgumentType]
    output_value = output.get_value()[0]  # pyright: ignore[reportOptionalSubscript]
    assert isinstance(output_value, InputClarification)
    assert output_value.user_guidance == "Need clarification after tool call"

    # Second execution - should call the tool
    return_clarification = False
    tool_node_called = False
    agent.new_clarifications = []
    output = await agent.execute_async()

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
        config=get_test_config(argument_clarifications_enabled=True),
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


@pytest.mark.asyncio
async def test_oneshot_agent_calls_clarification_tool_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the oneshot agent correctly calls the clarification tool when needed async."""
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

    async def tool_call(self, input, config) -> dict[str, Any]:  # noqa: A002, ANN001, ARG001
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

    monkeypatch.setattr(ToolNode, "ainvoke", tool_call)

    (plan, plan_run) = get_test_plan_run()
    agent = OneShotAgent(
        plan=plan,
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(argument_clarifications_enabled=True),
        agent_memory=InMemoryStorage(),
        tool=AdditionTool(),
    )

    output = await agent.execute_async()

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


@pytest.mark.asyncio
async def test_oneshot_agent_templates_values_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the oneshot agent correctly templates values before calling the tool async."""
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

    async def tool_call(self, input, config) -> dict[str, Any]:  # noqa: A002, ANN001, ARG001
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

    monkeypatch.setattr(ToolNode, "ainvoke", tool_call)

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

    output = await agent.execute_async()

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


@pytest.mark.asyncio
async def test_oneshot_model_fails_without_tool_async() -> None:
    """Test that the oneshot model fails without a tool (async version)."""
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
        await tool_calling_model.ainvoke({"messages": [], "step_inputs": []})


# Additional tests for OneShotToolCallingModel.ainvoke specifically
@pytest.mark.asyncio
async def test_oneshot_tool_calling_model_ainvoke_success() -> None:
    """Test that OneShotToolCallingModel.ainvoke works correctly."""
    (plan, plan_run) = get_test_plan_run()
    agent = OneShotAgent(
        plan=plan,
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=AdditionTool(),
    )
    tool_context = ToolRunContext(
        end_user=agent.end_user,
        plan=agent.plan,
        plan_run=agent.plan_run,
        config=agent.config,
        clarifications=agent.plan_run.get_clarifications_for_step(),
    )

    model_response = AIMessage(content="")
    model_response.tool_calls = [
        {
            "name": "Add_Tool",
            "type": "tool_call",
            "id": "call_123",
            "args": {
                "a": 5,
                "b": 3,
            },
        },
    ]
    mock_model = get_mock_generative_model(response=model_response)

    tool_calling_model = OneShotToolCallingModel(
        mock_model, [AdditionTool().to_langchain(ctx=tool_context)], agent, tool_context
    )

    result = await tool_calling_model.ainvoke({"messages": [], "step_inputs": []})

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0].tool_calls[0]["name"] == "Add_Tool"
    assert result["messages"][0].tool_calls[0]["args"]["a"] == 5
    assert result["messages"][0].tool_calls[0]["args"]["b"] == 3


@pytest.mark.asyncio
async def test_oneshot_tool_calling_model_ainvoke_with_execution_hooks() -> None:
    """Test that OneShotToolCallingModel.ainvoke handles execution hooks correctly."""
    (plan, plan_run) = get_test_plan_run()

    mock_before_tool_call = mock.MagicMock(return_value=None)
    agent = OneShotAgent(
        plan=plan,
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=AdditionTool(),
        execution_hooks=ExecutionHooks(before_tool_call=mock_before_tool_call),
    )
    tool_context = ToolRunContext(
        end_user=agent.end_user,
        plan=agent.plan,
        plan_run=agent.plan_run,
        config=agent.config,
        clarifications=agent.plan_run.get_clarifications_for_step(),
    )

    model_response = AIMessage(content="")
    model_response.tool_calls = [
        {
            "name": "Add_Tool",
            "type": "tool_call",
            "id": "call_123",
            "args": {
                "a": 5,
                "b": 3,
            },
        },
    ]
    mock_model = get_mock_generative_model(response=model_response)

    tool_calling_model = OneShotToolCallingModel(
        mock_model, [AdditionTool().to_langchain(ctx=tool_context)], agent, tool_context
    )

    result = await tool_calling_model.ainvoke({"messages": [], "step_inputs": []})

    # Verify that before_tool_call was called
    mock_before_tool_call.assert_called_once()
    assert "messages" in result
    assert len(result["messages"]) == 1


@pytest.mark.asyncio
async def test_oneshot_tool_calling_model_ainvoke_with_clarification_return() -> None:
    """Test OneShotToolCallingModel.ainvoke with clarification return."""
    (plan, plan_run) = get_test_plan_run()

    def before_tool_call(tool, args, plan_run, step) -> InputClarification:  # noqa: ANN001, ARG001
        return InputClarification(
            plan_run_id=plan_run.id,
            user_guidance="Need clarification",
            step=plan_run.current_step_index,
            argument_name="test_arg",
            source="Test",
        )

    agent = OneShotAgent(
        plan=plan,
        plan_run=plan_run,
        end_user=EndUser(external_id="123"),
        config=get_test_config(),
        agent_memory=InMemoryStorage(),
        tool=AdditionTool(),
        execution_hooks=ExecutionHooks(before_tool_call=before_tool_call),
    )
    tool_context = ToolRunContext(
        end_user=agent.end_user,
        plan=agent.plan,
        plan_run=agent.plan_run,
        config=agent.config,
        clarifications=agent.plan_run.get_clarifications_for_step(),
    )

    model_response = AIMessage(content="")
    model_response.tool_calls = [
        {
            "name": "Add_Tool",
            "type": "tool_call",
            "id": "call_123",
            "args": {
                "a": 5,
                "b": 3,
            },
        },
    ]
    mock_model = get_mock_generative_model(response=model_response)

    tool_calling_model = OneShotToolCallingModel(
        mock_model, [AdditionTool().to_langchain(ctx=tool_context)], agent, tool_context
    )

    result = await tool_calling_model.ainvoke({"messages": [], "step_inputs": []})

    # Should return empty messages when clarification is returned
    assert "messages" in result
    assert result["messages"] == []
    assert len(agent.new_clarifications) == 1
    assert isinstance(agent.new_clarifications[0], InputClarification)
    assert agent.new_clarifications[0].argument_name == "test_arg"
