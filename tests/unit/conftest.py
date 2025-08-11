"""Shared fixtures for portia unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from portia.config import FEATURE_FLAG_AGENT_MEMORY_ENABLED, GenerativeModelsConfig
from portia.model import GenerativeModel
from portia.open_source_tools.llm_tool import LLMTool
from portia.portia import Portia
from portia.telemetry.telemetry_service import BaseProductTelemetry
from portia.tool_registry import ToolRegistry
from tests.utils import AdditionTool, ClarificationTool, get_test_config


@pytest.fixture
def telemetry() -> MagicMock:
    """Fixture to create ProductTelemetry mock."""
    return MagicMock(spec=BaseProductTelemetry)


@pytest.fixture
def planning_model() -> MagicMock:
    """Fixture to create a mock planning model."""
    return MagicMock(spec=GenerativeModel)


@pytest.fixture
def default_model() -> MagicMock:
    """Fixture to create a mock default model."""
    return MagicMock(spec=GenerativeModel)


@pytest.fixture
def portia(planning_model: MagicMock, default_model: MagicMock, telemetry: MagicMock) -> Portia:
    """Fixture to create a Portia instance for testing."""
    config = get_test_config(
        models=GenerativeModelsConfig(
            planning_model=planning_model,
            default_model=default_model,
        ),
    )
    tool_registry = ToolRegistry([AdditionTool(), ClarificationTool(), LLMTool()])
    return Portia(config=config, tools=tool_registry, telemetry=telemetry)


@pytest.fixture
def portia_with_agent_memory(
    planning_model: MagicMock, default_model: MagicMock, telemetry: MagicMock
) -> Portia:
    """Fixture to create a Portia instance for testing with agent memory enabled."""
    config = get_test_config(
        # Set a small threshold value so all outputs are stored in agent memory
        feature_flags={FEATURE_FLAG_AGENT_MEMORY_ENABLED: True},
        large_output_threshold_tokens=3,
        models=GenerativeModelsConfig(
            planning_model=planning_model,
            default_model=default_model,
        ),
        portia_api_endpoint="https://api.portialabs.ai",
        portia_api_key="test-api-key",
    )
    tool_registry = ToolRegistry([AdditionTool(), ClarificationTool()])
    return Portia(config=config, tools=tool_registry, telemetry=telemetry)
