"""Test output."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from openai import BaseModel
from pydantic import HttpUrl

from portia.clarification import ActionClarification
from portia.config import LLMModel
from portia.execution_agents.output import AgentMemoryStorageDetails, Output
from portia.prefixed_uuid import PlanRunUUID


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


def test_local_output() -> None:
    """Test value is held locally."""
    output = Output(value="test value")
    assert output.value_for_prompt() == "test value"

    mock_agent_memory = MagicMock()
    assert output.full_value(mock_agent_memory) == "test value"
    mock_agent_memory.get_plan_run_output.assert_not_called()


def test_agent_memory_output() -> None:
    """Test value is stored in agent memory."""
    storage_details = AgentMemoryStorageDetails(
        name="test_value",
        plan_run_id=PlanRunUUID(),
    )
    output = Output(value=storage_details, summary="test summary")
    assert output.value_for_prompt() == "test summary"

    mock_agent_memory = MagicMock()
    mock_agent_memory.get_plan_run_output.return_value = "retrieved value"

    result = output.full_value(mock_agent_memory)
    assert result == "retrieved value"
    mock_agent_memory.get_plan_run_output.assert_called_once_with(
        storage_details.name,
        storage_details.plan_run_id,
    )
