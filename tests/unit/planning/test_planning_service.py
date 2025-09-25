"""Unit tests for the PlanningService class."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from portia.config import Config
from portia.end_user import EndUser
from portia.errors import PlanError, PlanNotFoundError, StorageError
from portia.plan import Plan, PlanContext, PlanInput, PlanUUID
from portia.planning.service import PlanningService
from portia.planning_agents.base_planning_agent import BasePlanningAgent
from portia.planning_agents.default_planning_agent import DefaultPlanningAgent
from portia.storage import InMemoryStorage
from portia.telemetry.telemetry_service import BaseProductTelemetry
from portia.tool import Tool
from portia.tool_registry import ToolRegistry


class MockPlanningAgent(BasePlanningAgent):
    """Mock planning agent for testing."""

    def __init__(self, outcome_error=None, steps=None):
        self.outcome_error = outcome_error
        self.steps = steps or []

    def generate_steps_or_error(self, **kwargs):
        result = MagicMock()
        result.error = self.outcome_error
        result.steps = self.steps
        return result

    async def agenerate_steps_or_error(self, **kwargs):
        result = MagicMock()
        result.error = self.outcome_error
        result.steps = self.steps
        return result


class MockOutputSchema(BaseModel):
    """Mock output schema for testing."""

    result: str


class TestPlanningService:
    """Test suite for PlanningService."""

    @pytest.fixture
    def config(self):
        """Create a mock config."""
        return Config.from_default()

    @pytest.fixture
    def tool_registry(self):
        """Create a mock tool registry."""
        registry = MagicMock(spec=ToolRegistry)
        tool = MagicMock(spec=Tool)
        tool.id = "test_tool"
        registry.match_tools.return_value = [tool]
        registry.get_tool.return_value = tool
        return registry

    @pytest.fixture
    def storage(self):
        """Create a mock storage."""
        storage = MagicMock(spec=InMemoryStorage)
        storage.get_plan_by_query.side_effect = StorageError("Plan not found")
        storage.aget_plan_by_query.side_effect = StorageError("Plan not found")
        storage.save_plan.return_value = None
        storage.asave_plan.return_value = None
        return storage

    @pytest.fixture
    def telemetry(self):
        """Create a mock telemetry."""
        return MagicMock(spec=BaseProductTelemetry)

    @pytest.fixture
    def planning_service(self, config, tool_registry, storage, telemetry):
        """Create a PlanningService instance."""
        return PlanningService(
            config=config,
            tool_registry=tool_registry,
            storage=storage,
            telemetry=telemetry,
        )

    def test_init(self, config, tool_registry, storage, telemetry):
        """Test PlanningService initialization."""
        service = PlanningService(
            config=config,
            tool_registry=tool_registry,
            storage=storage,
            telemetry=telemetry,
        )
        assert service.config == config
        assert service.tool_registry == tool_registry
        assert service.storage == storage
        assert service.telemetry == telemetry

    @patch("portia.planning.service.logger")
    def test_plan_success(self, mock_logger, planning_service):
        """Test successful plan generation."""
        # Mock planning agent
        steps = [MagicMock()]
        mock_agent = MockPlanningAgent(steps=steps)

        with patch.object(planning_service, "_get_planning_agent", return_value=mock_agent):
            result = planning_service.plan("test query")

        assert isinstance(result, Plan)
        assert result.plan_context.query == "test query"
        planning_service.telemetry.capture.assert_called_once()
        planning_service.storage.save_plan.assert_called_once()

    @patch("portia.planning.service.logger")
    async def test_aplan_success(self, mock_logger, planning_service):
        """Test successful async plan generation."""
        steps = [MagicMock()]
        mock_agent = MockPlanningAgent(steps=steps)

        with patch.object(planning_service, "_get_planning_agent", return_value=mock_agent):
            result = await planning_service.aplan("test query")

        assert isinstance(result, Plan)
        assert result.plan_context.query == "test query"
        planning_service.telemetry.capture.assert_called_once()
        planning_service.storage.asave_plan.assert_called_once()

    @patch("portia.planning.service.logger")
    async def test_generate_plan_async(self, mock_logger, planning_service):
        """Test generate_plan_async method."""
        steps = [MagicMock()]
        mock_agent = MockPlanningAgent(steps=steps)

        with patch.object(planning_service, "_get_planning_agent", return_value=mock_agent):
            result = await planning_service.generate_plan_async("test query")

        assert isinstance(result, Plan)
        assert result.plan_context.query == "test query"

    def test_plan_with_tools(self, planning_service):
        """Test plan generation with specified tools."""
        tool = MagicMock(spec=Tool)
        tool.id = "specific_tool"
        tools = [tool]

        steps = [MagicMock()]
        mock_agent = MockPlanningAgent(steps=steps)

        with patch.object(planning_service, "_get_planning_agent", return_value=mock_agent):
            result = planning_service.plan("test query", tools=tools)

        assert isinstance(result, Plan)

    def test_plan_with_tool_strings(self, planning_service):
        """Test plan generation with tool strings."""
        tools = ["tool1", "tool2"]

        steps = [MagicMock()]
        mock_agent = MockPlanningAgent(steps=steps)

        with patch.object(planning_service, "_get_planning_agent", return_value=mock_agent):
            result = planning_service.plan("test query", tools=tools)

        assert isinstance(result, Plan)
        # Verify tool registry was called to get tools
        assert planning_service.tool_registry.get_tool.call_count == len(tools)

    def test_plan_with_cached_plan(self, planning_service):
        """Test plan generation with cached plan."""
        cached_plan = Plan(
            plan_context=PlanContext(query="test query", tool_ids=[]),
            steps=[],
        )
        planning_service.storage.get_plan_by_query.return_value = cached_plan

        result = planning_service.plan("test query", use_cached_plan=True)

        assert result == cached_plan
        planning_service.storage.get_plan_by_query.assert_called_once_with("test query")

    async def test_aplan_with_cached_plan(self, planning_service):
        """Test async plan generation with cached plan."""
        cached_plan = Plan(
            plan_context=PlanContext(query="test query", tool_ids=[]),
            steps=[],
        )
        planning_service.storage.aget_plan_by_query.return_value = cached_plan

        result = await planning_service.aplan("test query", use_cached_plan=True)

        assert result == cached_plan
        planning_service.storage.aget_plan_by_query.assert_called_once_with("test query")

    def test_plan_with_end_user_string(self, planning_service):
        """Test plan generation with end user as string."""
        steps = [MagicMock()]
        mock_agent = MockPlanningAgent(steps=steps)

        with patch.object(planning_service, "_get_planning_agent", return_value=mock_agent):
            result = planning_service.plan("test query", end_user="user123")

        assert isinstance(result, Plan)

    def test_plan_with_end_user_object(self, planning_service):
        """Test plan generation with end user as EndUser object."""
        end_user = EndUser(external_id="user123", name="Test User")
        steps = [MagicMock()]
        mock_agent = MockPlanningAgent(steps=steps)

        with patch.object(planning_service, "_get_planning_agent", return_value=mock_agent):
            result = planning_service.plan("test query", end_user=end_user)

        assert isinstance(result, Plan)

    def test_plan_with_plan_inputs_dict_list(self, planning_service):
        """Test plan generation with plan inputs as dict list."""
        plan_inputs = [
            {"name": "input1", "description": "First input"},
            {"name": "input2"},
        ]
        steps = [MagicMock()]
        mock_agent = MockPlanningAgent(steps=steps)

        with patch.object(planning_service, "_get_planning_agent", return_value=mock_agent):
            result = planning_service.plan("test query", plan_inputs=plan_inputs)

        assert isinstance(result, Plan)
        assert len(result.plan_inputs) == 2

    def test_plan_with_plan_inputs_string_list(self, planning_service):
        """Test plan generation with plan inputs as string list."""
        plan_inputs = ["input1", "input2"]
        steps = [MagicMock()]
        mock_agent = MockPlanningAgent(steps=steps)

        with patch.object(planning_service, "_get_planning_agent", return_value=mock_agent):
            result = planning_service.plan("test query", plan_inputs=plan_inputs)

        assert isinstance(result, Plan)
        assert len(result.plan_inputs) == 2

    def test_plan_with_structured_output_schema(self, planning_service):
        """Test plan generation with structured output schema."""
        steps = [MagicMock()]
        mock_agent = MockPlanningAgent(steps=steps)

        with patch.object(planning_service, "_get_planning_agent", return_value=mock_agent):
            result = planning_service.plan(
                "test query", structured_output_schema=MockOutputSchema
            )

        assert isinstance(result, Plan)
        assert result.structured_output_schema == MockOutputSchema

    def test_plan_error_handling(self, planning_service):
        """Test plan generation error handling."""
        mock_agent = MockPlanningAgent(outcome_error="Planning failed")

        with patch.object(planning_service, "_get_planning_agent", return_value=mock_agent), \
             patch.object(planning_service, "_log_replan_with_portia_cloud_tools"):
            with pytest.raises(PlanError) as exc_info:
                planning_service.plan("test query")

            assert "Planning failed" in str(exc_info.value)

    async def test_aplan_error_handling(self, planning_service):
        """Test async plan generation error handling."""
        mock_agent = MockPlanningAgent(outcome_error="Planning failed")

        with patch.object(planning_service, "_get_planning_agent", return_value=mock_agent), \
             patch.object(planning_service, "_log_replan_with_portia_cloud_tools"):
            with pytest.raises(PlanError) as exc_info:
                await planning_service.aplan("test query")

            assert "Planning failed" in str(exc_info.value)

    def test_resolve_example_plans_none(self, planning_service):
        """Test resolve example plans with None input."""
        result = planning_service._resolve_example_plans(None)
        assert result is None

    def test_resolve_example_plans_with_plan_objects(self, planning_service):
        """Test resolve example plans with Plan objects."""
        plan1 = Plan(plan_context=PlanContext(query="query1", tool_ids=[]), steps=[])
        plan2 = Plan(plan_context=PlanContext(query="query2", tool_ids=[]), steps=[])

        result = planning_service._resolve_example_plans([plan1, plan2])

        assert result == [plan1, plan2]

    def test_resolve_example_plans_with_plan_uuids(self, planning_service):
        """Test resolve example plans with PlanUUID objects."""
        plan = Plan(plan_context=PlanContext(query="query", tool_ids=[]), steps=[])
        plan_uuid = plan.id
        planning_service.storage.get_plan.return_value = plan

        result = planning_service._resolve_example_plans([plan_uuid])

        assert result == [plan]
        planning_service.storage.get_plan.assert_called_once_with(plan_uuid)

    def test_resolve_example_plans_with_strings(self, planning_service):
        """Test resolve example plans with string IDs."""
        plan = Plan(plan_context=PlanContext(query="query", tool_ids=[]), steps=[])
        plan_id_str = str(plan.id)
        planning_service.storage.get_plan.return_value = plan

        result = planning_service._resolve_example_plans([plan_id_str])

        assert result == [plan]

    def test_resolve_example_plans_invalid_string(self, planning_service):
        """Test resolve example plans with invalid string."""
        with pytest.raises(ValueError) as exc_info:
            planning_service._resolve_example_plans(["invalid-string"])

        assert "must be a plan ID" in str(exc_info.value)

    def test_resolve_example_plans_invalid_type(self, planning_service):
        """Test resolve example plans with invalid type."""
        with pytest.raises(TypeError) as exc_info:
            planning_service._resolve_example_plans([123])

        assert "Invalid example plan type" in str(exc_info.value)

    async def test_aresolve_example_plans_none(self, planning_service):
        """Test async resolve example plans with None input."""
        result = await planning_service._aresolve_example_plans(None)
        assert result is None

    async def test_aresolve_example_plans_with_plan_objects(self, planning_service):
        """Test async resolve example plans with Plan objects."""
        plan1 = Plan(plan_context=PlanContext(query="query1", tool_ids=[]), steps=[])
        plan2 = Plan(plan_context=PlanContext(query="query2", tool_ids=[]), steps=[])

        result = await planning_service._aresolve_example_plans([plan1, plan2])

        assert result == [plan1, plan2]

    async def test_aresolve_example_plans_with_plan_uuids(self, planning_service):
        """Test async resolve example plans with PlanUUID objects."""
        plan = Plan(plan_context=PlanContext(query="query", tool_ids=[]), steps=[])
        plan_uuid = plan.id
        planning_service.storage.aget_plan.return_value = plan

        result = await planning_service._aresolve_example_plans([plan_uuid])

        assert result == [plan]
        planning_service.storage.aget_plan.assert_called_once_with(plan_uuid)

    def test_coerce_plan_inputs_none(self, planning_service):
        """Test coerce plan inputs with None."""
        result = planning_service._coerce_plan_inputs(None)
        assert result is None

    def test_coerce_plan_inputs_dict_list(self, planning_service):
        """Test coerce plan inputs with dict list."""
        plan_inputs = [
            {"name": "input1", "description": "First input"},
            {"name": "input2"},
        ]

        result = planning_service._coerce_plan_inputs(plan_inputs)

        assert len(result) == 2
        assert result[0].name == "input1"
        assert result[0].description == "First input"
        assert result[1].name == "input2"
        assert result[1].description is None

    def test_coerce_plan_inputs_string_list(self, planning_service):
        """Test coerce plan inputs with string list."""
        plan_inputs = ["input1", "input2"]

        result = planning_service._coerce_plan_inputs(plan_inputs)

        assert len(result) == 2
        assert result[0].name == "input1"
        assert result[1].name == "input2"

    def test_coerce_plan_inputs_plan_input_objects(self, planning_service):
        """Test coerce plan inputs with PlanInput objects."""
        plan_input = PlanInput(name="input1", description="desc")
        plan_inputs = [plan_input]

        result = planning_service._coerce_plan_inputs(plan_inputs)

        assert result == [plan_input]

    def test_coerce_plan_inputs_missing_name(self, planning_service):
        """Test coerce plan inputs with missing name."""
        plan_inputs = [{"description": "desc"}]

        with pytest.raises(ValueError) as exc_info:
            planning_service._coerce_plan_inputs(plan_inputs)

        assert "must have a name" in str(exc_info.value)

    def test_coerce_plan_inputs_invalid_type(self, planning_service):
        """Test coerce plan inputs with invalid type."""
        with pytest.raises(ValueError) as exc_info:
            planning_service._coerce_plan_inputs("invalid")

        assert "Invalid plan inputs received" in str(exc_info.value)

    def test_initialize_end_user_string(self, planning_service):
        """Test initialize end user with string."""
        result = planning_service._initialize_end_user("user123")

        assert isinstance(result, EndUser)
        assert result.external_id == "user123"

    def test_initialize_end_user_empty_string(self, planning_service):
        """Test initialize end user with empty string."""
        result = planning_service._initialize_end_user("")

        assert isinstance(result, EndUser)
        assert result.external_id == "portia:default_user"

    def test_initialize_end_user_object(self, planning_service):
        """Test initialize end user with EndUser object."""
        end_user = EndUser(external_id="user123", name="Test User")
        result = planning_service._initialize_end_user(end_user)

        assert result == end_user

    def test_initialize_end_user_none(self, planning_service):
        """Test initialize end user with None."""
        result = planning_service._initialize_end_user(None)

        assert isinstance(result, EndUser)
        assert result.external_id == "portia:default_user"

    def test_initialize_end_user_invalid_type(self, planning_service):
        """Test initialize end user with invalid type."""
        with pytest.raises(ValueError) as exc_info:
            planning_service._initialize_end_user(123)

        assert "end_user must be a string, EndUser, or None" in str(exc_info.value)

    async def test_ainitialize_end_user_string(self, planning_service):
        """Test async initialize end user with string."""
        result = await planning_service._ainitialize_end_user("user123")

        assert isinstance(result, EndUser)
        assert result.external_id == "user123"

    def test_load_plan_by_uuid_success(self, planning_service):
        """Test load plan by UUID success."""
        plan = Plan(plan_context=PlanContext(query="query", tool_ids=[]), steps=[])
        plan_uuid = plan.id
        planning_service.storage.get_plan.return_value = plan

        result = planning_service._load_plan_by_uuid(plan_uuid)

        assert result == plan

    def test_load_plan_by_uuid_not_found(self, planning_service):
        """Test load plan by UUID when not found."""
        plan_uuid = PlanUUID.new()
        planning_service.storage.get_plan.side_effect = Exception("Not found")

        with pytest.raises(PlanNotFoundError):
            planning_service._load_plan_by_uuid(plan_uuid)

    async def test_aload_plan_by_uuid_success(self, planning_service):
        """Test async load plan by UUID success."""
        plan = Plan(plan_context=PlanContext(query="query", tool_ids=[]), steps=[])
        plan_uuid = plan.id
        planning_service.storage.aget_plan.return_value = plan

        result = await planning_service._aload_plan_by_uuid(plan_uuid)

        assert result == plan

    async def test_aload_plan_by_uuid_not_found(self, planning_service):
        """Test async load plan by UUID when not found."""
        plan_uuid = PlanUUID.new()
        planning_service.storage.aget_plan.side_effect = Exception("Not found")

        with pytest.raises(PlanNotFoundError):
            await planning_service._aload_plan_by_uuid(plan_uuid)

    def test_get_planning_agent_default(self, planning_service):
        """Test get planning agent with default type."""
        agent = planning_service._get_planning_agent()
        assert isinstance(agent, DefaultPlanningAgent)

    def test_log_replan_with_portia_cloud_tools_no_api_key(self, planning_service, config):
        """Test log replan with no API key."""
        # This method should return early if API key is not set
        planning_service._log_replan_with_portia_cloud_tools(
            "error", "query", EndUser(external_id="user")
        )
        # Should not raise any exceptions

    @patch("portia.planning.service.PortiaCloudClient")
    @patch("portia.planning.service.PortiaToolRegistry")
    def test_log_replan_with_portia_cloud_tools_success(
        self, mock_registry_class, mock_client_class, planning_service
    ):
        """Test log replan with successful cloud tools."""
        # Mock the cloud client and registry
        mock_client = MagicMock()
        mock_client_class.new_client.return_value = mock_client

        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry
        mock_registry.with_default_tool_filter.return_value = mock_registry

        # Mock the combined registry
        mock_combined = MagicMock()
        planning_service.tool_registry.__add__ = MagicMock(return_value=mock_combined)
        mock_combined.match_tools.return_value = []

        # Mock successful planning outcome
        mock_agent = MockPlanningAgent(outcome_error=None, steps=[])

        with patch.object(planning_service, "_get_planning_agent", return_value=mock_agent):
            with pytest.raises(PlanError):
                planning_service._log_replan_with_portia_cloud_tools(
                    "original error", "query", EndUser(external_id="user")
                )
