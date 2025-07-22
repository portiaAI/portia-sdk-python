"""Test simple agent."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import ToolMessage
from langgraph.graph import END, MessagesState
from openai import BaseModel
from pydantic import HttpUrl

from portia.clarification import ActionClarification, InputClarification
from portia.config import FEATURE_FLAG_AGENT_MEMORY_ENABLED, LLMModel
from portia.end_user import EndUser
from portia.execution_agents.base_execution_agent import MAX_RETRIES, BaseExecutionAgent
from portia.execution_agents.context import StepInput
from portia.execution_agents.execution_utils import AgentNode
from portia.execution_agents.output import LocalDataValue
from portia.execution_hooks import ExecutionHooks
from portia.prefixed_uuid import PlanRunUUID
from portia.storage import InMemoryStorage
from tests.utils import AdditionTool, get_test_config, get_test_plan_run, get_test_tool_context


def test_base_agent_default_context() -> None:
    """Test default context."""
    plan, plan_run = get_test_plan_run()
    agent = BaseExecutionAgent(
        plan,
        plan_run,
        get_test_config(),
        EndUser(external_id="test"),
        InMemoryStorage(),
        None,
    )
    context = agent.get_system_context(
        get_test_tool_context(),
        [
            StepInput(
                name="$output1",
                value="test1",
                full_value="test1",
                description="Previous output 1",
            )
        ],
    )
    assert context is not None
    assert "test1" in context


def test_output_serialize() -> None:
    """Test output serialize."""

    class MyModel(BaseModel):
        id: str

    class NotAModel:
        id: str

        def __init__(self, id: str) -> None:  # noqa: A002
            self.id = id

    not_a_model = NotAModel(id="123")
    now = datetime.now(tz=UTC)
    clarification = ActionClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="",
        action_url=HttpUrl("https://example.com"),
        source="Test base execution agent",
    )

    tcs: list[tuple[Any, Any]] = [
        ("Hello World!", "Hello World!"),
        (None, ""),
        ({"hello": "world"}, json.dumps({"hello": "world"})),
        ([{"hello": "world"}], json.dumps([{"hello": "world"}])),
        (("hello", "world"), json.dumps(["hello", "world"])),
        ({"hello"}, json.dumps(["hello"])),  # sets don't have ordering
        (1, "1"),
        (1.23, "1.23"),
        (False, "false"),
        (LLMModel.GPT_4_O, str(LLMModel.GPT_4_O.value)),
        (MyModel(id="123"), MyModel(id="123").model_dump_json()),
        (b"Hello World!", "Hello World!"),
        (now, now.isoformat()),
        (not_a_model, str(not_a_model)),
        ([clarification], json.dumps([clarification.model_dump(mode="json")])),
    ]

    for tc in tcs:
        output = LocalDataValue(value=tc[0]).serialize_value()
        assert output == tc[1]


def test_next_state_after_tool_call_no_error() -> None:
    """Test next state when tool call succeeds."""
    execution_hooks = ExecutionHooks(
        after_tool_call=MagicMock(),
    )
    plan, plan_run = get_test_plan_run()
    agent = BaseExecutionAgent(
        plan,
        plan_run,
        get_test_config(),
        EndUser(external_id="test"),
        InMemoryStorage(),
        AdditionTool(),
        execution_hooks,
    )

    messages: list[ToolMessage] = [
        ToolMessage(
            content="Success message",
            tool_call_id="123",
            name="test_tool",
        ),
    ]
    state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

    result = agent.next_state_after_tool_call(agent.config, state, agent.tool)

    assert result == END
    assert execution_hooks.after_tool_call.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]


def test_next_state_after_tool_call_with_summarize() -> None:
    """Test next state when tool call succeeds and should summarize."""
    plan, plan_run = get_test_plan_run()
    tool = AdditionTool()
    tool.should_summarize = True

    agent = BaseExecutionAgent(
        plan,
        plan_run,
        get_test_config(),
        EndUser(external_id="test"),
        InMemoryStorage(),
        tool,
    )

    messages: list[ToolMessage] = [
        ToolMessage(
            content="Success message",
            tool_call_id="123",
            name="test_tool",
        ),
    ]
    state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

    result = agent.next_state_after_tool_call(agent.config, state, tool)

    assert result == AgentNode.SUMMARIZER


def test_next_state_after_tool_call_with_large_output() -> None:
    """Test next state when tool call succeeds and should summarize."""
    plan, plan_run = get_test_plan_run()
    tool = AdditionTool()

    agent = BaseExecutionAgent(
        plan,
        plan_run,
        get_test_config(
            # Set a small threshold value so all outputs are stored in agent memory
            feature_flags={FEATURE_FLAG_AGENT_MEMORY_ENABLED: True},
            large_output_threshold_tokens=10,
        ),
        EndUser(external_id="test"),
        InMemoryStorage(),
        tool,
    )

    messages: list[ToolMessage] = [
        ToolMessage(
            content="Test" * 1000,
            tool_call_id="123",
            name="test_tool",
        ),
    ]
    state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

    result = agent.next_state_after_tool_call(agent.config, state, tool)
    assert result == AgentNode.SUMMARIZER


def test_next_state_after_tool_call_with_error_retry() -> None:
    """Test next state when tool call fails and max retries reached."""
    plan, plan_run = get_test_plan_run()
    tool = AdditionTool()

    agent = BaseExecutionAgent(
        plan,
        plan_run,
        get_test_config(),
        EndUser(external_id="test"),
        InMemoryStorage(),
        tool,
    )

    for i in range(1, MAX_RETRIES + 1):
        messages: list[ToolMessage] = [
            ToolMessage(
                content=f"ToolSoftError: Error {j}",
                tool_call_id=str(j),
                name="test_tool",
            )
            for j in range(1, i + 1)
        ]
        state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

        result = agent.next_state_after_tool_call(agent.config, state)

        expected_state = END if i == MAX_RETRIES else AgentNode.TOOL_AGENT
        assert result == expected_state, f"Failed at retry {i}"


def test_next_state_after_tool_call_with_clarification_artifact() -> None:
    """Test next state when tool call succeeds with clarification artifact."""
    plan, plan_run = get_test_plan_run()
    tool = AdditionTool()
    tool.should_summarize = True

    agent = BaseExecutionAgent(
        plan,
        plan_run,
        get_test_config(),
        EndUser(external_id="test"),
        InMemoryStorage(),
        tool,
    )

    clarification = InputClarification(
        argument_name="test",
        user_guidance="test",
        plan_run_id=PlanRunUUID(),
        source="Test base execution agent",
    )

    messages: list[ToolMessage] = [
        ToolMessage(
            content="Success message",
            tool_call_id="123",
            name="test_tool",
            artifact=clarification,
        ),
    ]
    state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

    result = agent.next_state_after_tool_call(agent.config, state, tool)

    # Should return END even though tool.should_summarize is True
    # because the message contains a clarification artifact
    assert result == END


def test_next_state_after_tool_call_with_list_of_clarifications() -> None:
    """Test next state when tool call succeeds with a list of clarifications as artifact."""
    plan, plan_run = get_test_plan_run()
    tool = AdditionTool()
    tool.should_summarize = True

    agent = BaseExecutionAgent(
        plan,
        plan_run,
        get_test_config(),
        EndUser(external_id="test"),
        InMemoryStorage(),
        tool,
    )

    clarifications = [
        InputClarification(
            argument_name="test1",
            user_guidance="guidance1",
            plan_run_id=PlanRunUUID(),
            source="Test base execution agent",
        ),
        InputClarification(
            argument_name="test2",
            user_guidance="guidance2",
            plan_run_id=PlanRunUUID(),
            source="Test base execution agent",
        ),
    ]

    messages: list[ToolMessage] = [
        ToolMessage(
            content="Success message",
            tool_call_id="123",
            name="test_tool",
            artifact=clarifications,
        ),
    ]
    state: MessagesState = {"messages": messages}  # type: ignore  # noqa: PGH003

    result = agent.next_state_after_tool_call(agent.config, state, tool)

    # Should return END even though tool.should_summarize is True
    # because the message contains a list of clarifications as artifact
    assert result == END
