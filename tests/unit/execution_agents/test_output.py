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
from portia.execution_agents.output import AgentMemoryValue, LocalDataValue
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
    source="Test execution agents output",
)


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        pytest.param("Hello World!", "Hello World!", id="string"),
        pytest.param(None, "", id="none"),
        pytest.param({"hello": "world"}, json.dumps({"hello": "world"}), id="dict"),
        pytest.param([{"hello": "world"}], json.dumps([{"hello": "world"}]), id="list"),
        pytest.param(("hello", "world"), json.dumps(["hello", "world"]), id="tuple"),
        pytest.param({"hello"}, json.dumps(["hello"]), id="set"),  # sets don't have ordering
        pytest.param(1, "1", id="int"),
        pytest.param(1.23, "1.23", id="float"),
        pytest.param(False, "false", id="bool"),
        pytest.param(LLMModel.GPT_4_O, str(LLMModel.GPT_4_O.value), id="enum"),
        pytest.param(MyModel(id="123"), MyModel(id="123").model_dump_json(), id="model"),
        pytest.param(b"Hello World!", "Hello World!", id="bytes"),
        pytest.param(now, now.isoformat(), id="datetime"),
        pytest.param(not_a_model, str(not_a_model), id="not_a_model"),
        pytest.param(
            [clarification],
            json.dumps([clarification.model_dump(mode="json")]),
            id="list_of_clarification",
        ),
    ],
)
def test_output_serialize(input_value: Any, expected: Any) -> None:  # noqa: ANN401
    """Test output serialize."""
    output = LocalDataValue(value=input_value).serialize_value()
    assert output == expected


def test_local_output() -> None:
    """Test value is held locally."""
    output = LocalDataValue(value="test value")
    assert output.get_value() == "test value"

    mock_agent_memory = MagicMock(spec=AgentMemory)
    assert output.full_value(mock_agent_memory) == "test value"
    mock_agent_memory.get_plan_run_output.assert_not_called()


def test_agent_memory_output() -> None:
    """Test value is stored in agent memory."""
    output = AgentMemoryValue(
        output_name="test_value",
        plan_run_id=PlanRunUUID(),
        summary="test summary",
    )
    assert output.get_value() == "test summary"
    assert output.get_summary() == "test summary"
    assert output.serialize_value() == "test summary"

    mock_agent_memory = MagicMock()
    mock_agent_memory.get_plan_run_output.return_value = LocalDataValue(value="retrieved value")

    result = output.full_value(mock_agent_memory)
    assert result == "retrieved value"
    mock_agent_memory.get_plan_run_output.assert_called_once_with(
        output.output_name,
        output.plan_run_id,
    )
