"""Test output."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from openai import BaseModel
from pydantic import HttpUrl

from portia.clarification import ActionClarification
from portia.config import LLMModel
from portia.execution_agents.output import AgentMemoryOutput, LocalOutput
from portia.prefixed_uuid import PlanRunUUID
from portia.storage import AgentMemory


class MyModel(BaseModel):
    """Test BaseModel."""

    id: str


class NotAModel:
    """Test class that's not a BaseModel."""

    id: str

    def __init__(self, id: str) -> None:  # noqa: A002
        """Init an instance."""
        self.id = id


not_a_model = NotAModel(id="123")
now = datetime.now(tz=UTC)
clarification = ActionClarification(
    plan_run_id=PlanRunUUID(),
    user_guidance="",
    action_url=HttpUrl("https://example.com"),
)


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
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
    ],
)
def test_output_serialize(input_value: Any, expected: Any) -> None:  # noqa: ANN401
    """Test output serialize."""
    output = LocalOutput(value=input_value).serialize_value()
    assert output == expected


def test_local_output() -> None:
    """Test value is held locally."""
    output = LocalOutput(value="test value")
    assert output.get_value() == "test value"

    mock_agent_memory = MagicMock(spec=AgentMemory)
    assert output.full_value(mock_agent_memory) == "test value"
    mock_agent_memory.get_plan_run_output.assert_not_called()


def test_agent_memory_output() -> None:
    """Test value is stored in agent memory."""
    output = AgentMemoryOutput(
        output_name="test_value",
        plan_run_id=PlanRunUUID(),
        summary="test summary",
    )
    assert output.get_value() == "test summary"
    assert output.summary == "test summary"

    mock_agent_memory = MagicMock()
    mock_agent_memory.get_plan_run_output.return_value = "retrieved value"

    result = output.full_value(mock_agent_memory)
    assert result == "retrieved value"
    mock_agent_memory.get_plan_run_output.assert_called_once_with(
        output.output_name,
        output.plan_run_id,
    )
