"""Test simple agent."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from openai import BaseModel
from pydantic import HttpUrl

from portia.clarification import ActionClarification
from portia.config import LLMModel
from portia.execution_agents.base_execution_agent import BaseExecutionAgent
from portia.execution_agents.output import Output
from portia.execution_context import execution_context
from portia.prefixed_uuid import PlanRunUUID
from tests.utils import get_test_config, get_test_plan_run


def test_base_agent_default_context() -> None:
    """Test default context."""
    plan, plan_run = get_test_plan_run()
    agent = BaseExecutionAgent(
        plan.steps[0],
        plan_run,
        get_test_config(),
        None,
    )
    context = agent.get_system_context()
    assert context is not None
    assert "value: 1" in context


def test_base_agent_default_context_with_extensions() -> None:
    """Test default context."""
    plan, plan_run = get_test_plan_run()
    agent = BaseExecutionAgent(
        plan.steps[0],
        plan_run,
        get_test_config(),
        None,
    )
    with execution_context(agent_system_context_extension=["456"]):
        context = agent.get_system_context()
    assert context is not None
    assert "456" in context


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
        output = Output(value=tc[0]).serialize_value(tc[0])
        assert output == tc[1]
