"""Fixtures for open source tools."""

import uuid
from unittest.mock import MagicMock

import pytest

from portia.config import Config
from portia.model import GenerativeModel
from portia.prefixed_uuid import PlanRunUUID
from portia.tool import ToolRunContext


@pytest.fixture
def mock_tool_run_context(mock_model: MagicMock) -> ToolRunContext:
    """Fixture to mock ExecutionContext."""
    mock_config = MagicMock(spec=Config)
    mock_config.get_default_model.return_value = mock_model
    mock_config.get_generative_model.return_value = mock_model
    return ToolRunContext.model_construct(
        plan_run_id=PlanRunUUID(uuid=uuid.uuid4()),
        config=mock_config,
        clarifications=[],
    )


@pytest.fixture(autouse=True)
def mock_openai_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture to set the OPENAI_API_KEY environment variable."""
    monkeypatch.setenv("OPENAI_API_KEY", "123")


@pytest.fixture
def mock_model() -> MagicMock:
    """Fixture to mock a GenerativeModel."""
    return MagicMock(spec=GenerativeModel)
