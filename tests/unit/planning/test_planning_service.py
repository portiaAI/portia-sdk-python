"""Unit tests for PlanningService."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from portia.config import Config, PlanningAgentType
from portia.context import PortiaContext
from portia.end_user import EndUser
from portia.errors import PlanError, PlanNotFoundError
from portia.plan import Plan, PlanContext, PlanInput, PlanUUID, Step
from portia.planning.service import PlanningService
from portia.planning_agents.base_planning_agent import StepsOrError
from portia.planning_agents.default_planning_agent import DefaultPlanningAgent
from portia.storage import InMemoryStorage, StorageError
from portia.telemetry.telemetry_service import BaseProductTelemetry
from portia.tool import Tool
from portia.tool_registry import ToolRegistry
from tests.utils import AdditionTool, get_test_config

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture
def mock_storage() -> MagicMock:
    """Create a mock storage instance."""
    storage = MagicMock(spec=InMemoryStorage)
    storage.get_plan_by_query = MagicMock()
    storage.aget_plan_by_query = AsyncMock()
    storage.save_plan = MagicMock()
    storage.asave_plan = AsyncMock()
    storage.get_plan = MagicMock()
    storage.aget_plan = AsyncMock()
    storage.get_end_user = MagicMock(return_value=None)
    storage.aget_end_user = AsyncMock(return_value=None)
    storage.save_end_user = MagicMock()
    storage.asave_end_user = AsyncMock()
    return storage


@pytest.fixture
def mock_tool_registry() -> MagicMock:
    """Create a mock tool registry."""
    registry = MagicMock(spec=ToolRegistry)
    registry.match_tools = MagicMock()
    registry.get_tool = MagicMock()
    return registry


@pytest.fixture
def mock_config() -> Config:
    """Create a test config."""
    return get_test_config()


@pytest.fixture
def mock_telemetry() -> MagicMock:
    """Create a mock telemetry service."""
    return MagicMock(spec=BaseProductTelemetry)


@pytest.fixture
def portia_context(
    mock_storage: MagicMock,
    mock_tool_registry: MagicMock,
    mock_config: Config,
    mock_telemetry: MagicMock,
) -> PortiaContext:
    """Create a PortiaContext for testing."""
    return PortiaContext(
        storage=mock_storage,
        tool_registry=mock_tool_registry,
        config=mock_config,
        telemetry=mock_telemetry,
    )


@pytest.fixture
def planning_service(portia_context: PortiaContext) -> PlanningService:
    """Create a PlanningService instance for testing."""
    return PlanningService(context=portia_context)


def test_planning_service_init(portia_context: PortiaContext) -> None:
    """Test PlanningService initialization."""
    service = PlanningService(context=portia_context)
    assert service.context == portia_context


def test_generate_plan_with_cached_plan(
    planning_service: PlanningService, mock_storage: MagicMock
) -> None:
    """Test generate_plan returns cached plan when use_cached_plan is True."""
    cached_plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=["tool1"]),
        steps=[],
    )
    mock_storage.get_plan_by_query.return_value = cached_plan

    result = planning_service.generate_plan(
        query="test query",
        use_cached_plan=True,
    )

    assert result == cached_plan
    mock_storage.get_plan_by_query.assert_called_once_with("test query")


def test_generate_plan_with_cached_plan_fallback(
    planning_service: PlanningService,
    mock_storage: MagicMock,
    mock_tool_registry: MagicMock,
) -> None:
    """Test generate_plan falls back to new plan if cached plan fails."""
    mock_storage.get_plan_by_query.side_effect = StorageError("Not found")
    mock_tool_registry.match_tools.return_value = [AdditionTool()]

    # Mock the planning agent
    with patch.object(
        planning_service, "_get_planning_agent"
    ) as mock_get_agent:
        mock_agent = MagicMock(spec=DefaultPlanningAgent)
        mock_agent.generate_steps_or_error.return_value = StepsOrError(
            steps=[Step(task="test task", tool_id="tool1", output="$output")],
            error=None,
        )
        mock_get_agent.return_value = mock_agent

        # Mock end user initialization
        mock_end_user = EndUser(external_id="test_user")
        mock_storage.save_end_user.return_value = mock_end_user

        result = planning_service.generate_plan(
            query="test query",
            use_cached_plan=True,
        )

        assert isinstance(result, Plan)
        assert result.plan_context.query == "test query"


def test_generate_plan_with_explicit_tools(
    planning_service: PlanningService,
    mock_storage: MagicMock,
    mock_tool_registry: MagicMock,
) -> None:
    """Test generate_plan with explicit list of tools."""
    tool = AdditionTool()
    mock_tool_registry.get_tool.return_value = tool

    # Mock the planning agent
    with patch.object(
        planning_service, "_get_planning_agent"
    ) as mock_get_agent:
        mock_agent = MagicMock(spec=DefaultPlanningAgent)
        mock_agent.generate_steps_or_error.return_value = StepsOrError(
            steps=[Step(task="test task", tool_id="addition_tool", output="$output")],
            error=None,
        )
        mock_get_agent.return_value = mock_agent

        # Mock end user initialization
        mock_end_user = EndUser(external_id="test_user")
        mock_storage.save_end_user.return_value = mock_end_user

        result = planning_service.generate_plan(
            query="test query",
            tools=["addition_tool"],
        )

        assert isinstance(result, Plan)
        assert len(result.steps) == 1


def test_generate_plan_with_tool_objects(
    planning_service: PlanningService,
    mock_storage: MagicMock,
) -> None:
    """Test generate_plan with Tool objects."""
    tool = AdditionTool()

    # Mock the planning agent
    with patch.object(
        planning_service, "_get_planning_agent"
    ) as mock_get_agent:
        mock_agent = MagicMock(spec=DefaultPlanningAgent)
        mock_agent.generate_steps_or_error.return_value = StepsOrError(
            steps=[Step(task="test task", tool_id="add_tool", output="$output")],
            error=None,
        )
        mock_get_agent.return_value = mock_agent

        # Mock end user initialization
        mock_end_user = EndUser(external_id="test_user")
        mock_storage.save_end_user.return_value = mock_end_user

        result = planning_service.generate_plan(
            query="test query",
            tools=[tool],
        )

        assert isinstance(result, Plan)
        assert tool.id in result.plan_context.tool_ids


def test_generate_plan_raises_plan_error_on_agent_error(
    planning_service: PlanningService,
    mock_storage: MagicMock,
    mock_tool_registry: MagicMock,
) -> None:
    """Test generate_plan raises PlanError when agent returns error."""
    mock_tool_registry.match_tools.return_value = [AdditionTool()]

    # Mock the planning agent to return an error
    with patch.object(
        planning_service, "_get_planning_agent"
    ) as mock_get_agent:
        mock_agent = MagicMock(spec=DefaultPlanningAgent)
        mock_agent.generate_steps_or_error.return_value = StepsOrError(
            steps=[],
            error="Test error message",
        )
        mock_get_agent.return_value = mock_agent

        # Mock end user initialization
        mock_end_user = EndUser(external_id="test_user")
        mock_storage.save_end_user.return_value = mock_end_user

        with pytest.raises(PlanError, match="Test error message"):
            planning_service.generate_plan(query="test query")


@pytest.mark.asyncio
async def test_generate_plan_async_with_cached_plan(
    planning_service: PlanningService, mock_storage: MagicMock
) -> None:
    """Test generate_plan_async returns cached plan when use_cached_plan is True."""
    cached_plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=["tool1"]),
        steps=[],
    )
    mock_storage.aget_plan_by_query.return_value = cached_plan

    result = await planning_service.generate_plan_async(
        query="test query",
        use_cached_plan=True,
    )

    assert result == cached_plan
    mock_storage.aget_plan_by_query.assert_called_once_with("test query")


@pytest.mark.asyncio
async def test_generate_plan_async(
    planning_service: PlanningService,
    mock_storage: MagicMock,
    mock_tool_registry: MagicMock,
) -> None:
    """Test generate_plan_async creates a new plan."""
    mock_tool_registry.match_tools.return_value = [AdditionTool()]

    # Mock the planning agent
    with patch.object(
        planning_service, "_get_planning_agent"
    ) as mock_get_agent:
        mock_agent = MagicMock(spec=DefaultPlanningAgent)
        mock_agent.agenerate_steps_or_error = AsyncMock(
            return_value=StepsOrError(
                steps=[Step(task="test task", tool_id="tool1", output="$output")],
                error=None,
            )
        )
        mock_get_agent.return_value = mock_agent

        # Mock end user initialization
        mock_end_user = EndUser(external_id="test_user")
        mock_storage.asave_end_user.return_value = mock_end_user

        result = await planning_service.generate_plan_async(query="test query")

        assert isinstance(result, Plan)
        assert result.plan_context.query == "test query"
        mock_storage.asave_plan.assert_called_once()


def test_coerce_plan_inputs_with_none(planning_service: PlanningService) -> None:
    """Test _coerce_plan_inputs returns None when input is None."""
    result = planning_service._coerce_plan_inputs(None)
    assert result is None


def test_coerce_plan_inputs_with_plan_input_objects(
    planning_service: PlanningService,
) -> None:
    """Test _coerce_plan_inputs with PlanInput objects."""
    plan_inputs = [PlanInput(name="input1", description="desc1")]
    result = planning_service._coerce_plan_inputs(plan_inputs)
    assert result == plan_inputs


def test_coerce_plan_inputs_with_dicts(planning_service: PlanningService) -> None:
    """Test _coerce_plan_inputs with dictionaries."""
    plan_inputs = [{"name": "input1", "description": "desc1"}]
    result = planning_service._coerce_plan_inputs(plan_inputs)
    assert len(result) == 1
    assert result[0].name == "input1"
    assert result[0].description == "desc1"


def test_coerce_plan_inputs_with_strings(planning_service: PlanningService) -> None:
    """Test _coerce_plan_inputs with string names."""
    plan_inputs = ["input1", "input2"]
    result = planning_service._coerce_plan_inputs(plan_inputs)
    assert len(result) == 2
    assert result[0].name == "input1"
    assert result[1].name == "input2"


def test_coerce_plan_inputs_invalid_dict_without_name(
    planning_service: PlanningService,
) -> None:
    """Test _coerce_plan_inputs raises ValueError for dict without name."""
    plan_inputs = [{"description": "desc1"}]
    with pytest.raises(ValueError, match="Plan input must have a name"):
        planning_service._coerce_plan_inputs(plan_inputs)


def test_resolve_example_plans_with_none(planning_service: PlanningService) -> None:
    """Test _resolve_example_plans returns None when input is None."""
    result = planning_service._resolve_example_plans(None)
    assert result is None


def test_resolve_example_plans_with_plan_objects(
    planning_service: PlanningService,
) -> None:
    """Test _resolve_example_plans with Plan objects."""
    plan = Plan(
        plan_context=PlanContext(query="test", tool_ids=["tool1"]),
        steps=[],
    )
    result = planning_service._resolve_example_plans([plan])
    assert result == [plan]


def test_resolve_example_plans_with_plan_uuid(
    planning_service: PlanningService, mock_storage: MagicMock
) -> None:
    """Test _resolve_example_plans with PlanUUID."""
    plan = Plan(
        plan_context=PlanContext(query="test", tool_ids=["tool1"]),
        steps=[],
    )
    mock_storage.get_plan.return_value = plan

    result = planning_service._resolve_example_plans([plan.id])
    assert len(result) == 1
    assert result[0] == plan


def test_resolve_example_plans_with_plan_id_string(
    planning_service: PlanningService, mock_storage: MagicMock
) -> None:
    """Test _resolve_example_plans with plan ID string."""
    plan = Plan(
        plan_context=PlanContext(query="test", tool_ids=["tool1"]),
        steps=[],
    )
    plan_id_str = str(plan.id)
    mock_storage.get_plan.return_value = plan

    result = planning_service._resolve_example_plans([plan_id_str])
    assert len(result) == 1
    assert result[0] == plan


def test_resolve_string_example_plan_invalid_format(
    planning_service: PlanningService,
) -> None:
    """Test _resolve_string_example_plan raises ValueError for non-plan-ID string."""
    with pytest.raises(ValueError, match="must be a plan ID"):
        planning_service._resolve_string_example_plan("invalid-string")


def test_resolve_single_example_plan_invalid_type(
    planning_service: PlanningService,
) -> None:
    """Test _resolve_single_example_plan raises TypeError for invalid type."""
    with pytest.raises(TypeError, match="Invalid example plan type"):
        planning_service._resolve_single_example_plan(123)  # type: ignore


def test_load_plan_by_uuid_not_found(
    planning_service: PlanningService, mock_storage: MagicMock
) -> None:
    """Test _load_plan_by_uuid raises PlanNotFoundError when plan not found."""
    plan_uuid = PlanUUID()
    mock_storage.get_plan.side_effect = Exception("Not found")

    with pytest.raises(PlanNotFoundError):
        planning_service._load_plan_by_uuid(plan_uuid)


def test_initialize_end_user_with_string(
    planning_service: PlanningService, mock_storage: MagicMock
) -> None:
    """Test _initialize_end_user with string external_id."""
    existing_user = EndUser(external_id="test_user")
    mock_storage.get_end_user.return_value = existing_user

    result = planning_service._initialize_end_user("test_user")
    assert result == existing_user
    mock_storage.get_end_user.assert_called_once_with(external_id="test_user")


def test_initialize_end_user_with_none(
    planning_service: PlanningService, mock_storage: MagicMock
) -> None:
    """Test _initialize_end_user with None creates default user."""
    new_user = EndUser(external_id="portia:default_user")
    mock_storage.save_end_user.return_value = new_user

    result = planning_service._initialize_end_user(None)
    assert result == new_user


def test_initialize_end_user_with_end_user_object(
    planning_service: PlanningService, mock_storage: MagicMock
) -> None:
    """Test _initialize_end_user with EndUser object."""
    end_user = EndUser(external_id="custom_user")
    mock_storage.save_end_user.return_value = end_user

    result = planning_service._initialize_end_user(end_user)
    assert result == end_user
    mock_storage.save_end_user.assert_called_once_with(end_user)


def test_get_planning_agent(planning_service: PlanningService) -> None:
    """Test _get_planning_agent returns correct agent type."""
    agent = planning_service._get_planning_agent()
    assert isinstance(agent, DefaultPlanningAgent)


def test_generate_plan_with_plan_inputs(
    planning_service: PlanningService,
    mock_storage: MagicMock,
    mock_tool_registry: MagicMock,
) -> None:
    """Test generate_plan with plan inputs."""
    mock_tool_registry.match_tools.return_value = [AdditionTool()]

    plan_inputs = [PlanInput(name="input1", description="Test input")]

    # Mock the planning agent
    with patch.object(
        planning_service, "_get_planning_agent"
    ) as mock_get_agent:
        mock_agent = MagicMock(spec=DefaultPlanningAgent)
        mock_agent.generate_steps_or_error.return_value = StepsOrError(
            steps=[Step(task="test task", tool_id="tool1", output="$output")],
            error=None,
        )
        mock_get_agent.return_value = mock_agent

        # Mock end user initialization
        mock_end_user = EndUser(external_id="test_user")
        mock_storage.save_end_user.return_value = mock_end_user

        result = planning_service.generate_plan(
            query="test query",
            plan_inputs=plan_inputs,
        )

        assert isinstance(result, Plan)
        assert result.plan_inputs == plan_inputs


def test_generate_plan_with_structured_output_schema(
    planning_service: PlanningService,
    mock_storage: MagicMock,
    mock_tool_registry: MagicMock,
) -> None:
    """Test generate_plan with structured output schema."""
    mock_tool_registry.match_tools.return_value = [AdditionTool()]

    class OutputSchema(BaseModel):
        result: int

    # Mock the planning agent
    with patch.object(
        planning_service, "_get_planning_agent"
    ) as mock_get_agent:
        mock_agent = MagicMock(spec=DefaultPlanningAgent)
        mock_agent.generate_steps_or_error.return_value = StepsOrError(
            steps=[Step(task="test task", tool_id="tool1", output="$output")],
            error=None,
        )
        mock_get_agent.return_value = mock_agent

        # Mock end user initialization
        mock_end_user = EndUser(external_id="test_user")
        mock_storage.save_end_user.return_value = mock_end_user

        result = planning_service.generate_plan(
            query="test query",
            structured_output_schema=OutputSchema,
        )

        assert isinstance(result, Plan)
        assert result.structured_output_schema == OutputSchema
