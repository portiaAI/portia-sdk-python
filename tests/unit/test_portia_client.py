"""Tests for PortiaClient façade."""

import pytest
from unittest.mock import Mock, patch

from portia.client import PortiaClient
from portia.config import Config
from portia.execution_hooks import ExecutionHooks
from portia.telemetry.telemetry_service import BaseProductTelemetry
from portia.tool_registry import ToolRegistry


class TestPortiaClient:
    """Test PortiaClient façade behavior."""

    def test_init_with_defaults(self):
        """Test PortiaClient initialization with default parameters."""
        client = PortiaClient()

        assert hasattr(client, "_legacy_portia")
        assert client._legacy_portia is not None
        assert client._planning_service_override is None
        assert client._plan_run_service_override is None
        assert client._execution_engine_override is None
        assert client._clarification_manager_override is None
        assert client._builder_plan_executor_override is None

    def test_init_with_service_overrides(self):
        """Test PortiaClient initialization with service overrides."""
        mock_planning = Mock()
        mock_plan_run = Mock()
        mock_execution = Mock()
        mock_clarification = Mock()
        mock_builder = Mock()

        client = PortiaClient(
            planning_service=mock_planning,
            plan_run_service=mock_plan_run,
            execution_engine=mock_execution,
            clarification_manager=mock_clarification,
            builder_plan_executor=mock_builder,
        )

        assert client._planning_service_override is mock_planning
        assert client._plan_run_service_override is mock_plan_run
        assert client._execution_engine_override is mock_execution
        assert client._clarification_manager_override is mock_clarification
        assert client._builder_plan_executor_override is mock_builder

    def test_init_with_portia_params(self):
        """Test PortiaClient initialization with Portia parameters."""
        config = Config.from_default()
        tools = ToolRegistry([])
        hooks = ExecutionHooks()
        telemetry = Mock(spec=BaseProductTelemetry)

        with patch("portia.client._LegacyPortia") as mock_portia_class:
            mock_portia = Mock()
            mock_portia_class.return_value = mock_portia

            client = PortiaClient(
                config=config,
                tools=tools,
                execution_hooks=hooks,
                telemetry=telemetry,
            )

            mock_portia_class.assert_called_once_with(
                config=config,
                tools=tools,
                execution_hooks=hooks,
                telemetry=telemetry,
            )
            assert client._legacy_portia is mock_portia

    def test_lazy_service_properties(self):
        """Test lazy service instantiation properties."""
        client = PortiaClient()

        # Services should be None initially
        assert client._planning_service is None
        assert client._plan_run_service is None
        assert client._execution_engine is None
        assert client._clarification_manager is None
        assert client._builder_plan_executor is None

    def test_lazy_service_properties_with_overrides(self):
        """Test lazy service properties with overrides."""
        mock_planning = Mock()
        mock_plan_run = Mock()
        mock_execution = Mock()
        mock_clarification = Mock()
        mock_builder = Mock()

        client = PortiaClient(
            planning_service=mock_planning,
            plan_run_service=mock_plan_run,
            execution_engine=mock_execution,
            clarification_manager=mock_clarification,
            builder_plan_executor=mock_builder,
        )

        # Should return overrides
        assert client._planning_service is mock_planning
        assert client._plan_run_service is mock_plan_run
        assert client._execution_engine is mock_execution
        assert client._clarification_manager is mock_clarification
        assert client._builder_plan_executor is mock_builder

    def test_public_api_delegation(self):
        """Test that public API methods delegate to legacy Portia."""
        with patch("portia.client._LegacyPortia") as mock_portia_class:
            mock_portia = Mock()
            mock_portia_class.return_value = mock_portia

            client = PortiaClient()

            # Test run method delegation
            mock_portia.run.return_value = "run_result"
            result = client.run("test query")
            mock_portia.run.assert_called_once_with(
                query="test query",
                tools=None,
                example_plans=None,
                end_user=None,
                plan_run_inputs=None,
                structured_output_schema=None,
                use_cached_plan=False,
            )
            assert result == "run_result"

    @pytest.mark.asyncio
    async def test_async_api_delegation(self):
        """Test that async API methods delegate to legacy Portia."""
        from unittest.mock import AsyncMock

        with patch("portia.client._LegacyPortia") as mock_portia_class:
            mock_portia = Mock()
            mock_portia.arun = AsyncMock(return_value="arun_result")
            mock_portia_class.return_value = mock_portia

            client = PortiaClient()

            # Test arun method delegation
            result = await client.arun("test query")
            mock_portia.arun.assert_called_once_with(
                query="test query",
                tools=None,
                example_plans=None,
                end_user=None,
                plan_run_inputs=None,
                structured_output_schema=None,
                use_cached_plan=False,
            )
            assert result == "arun_result"

    def test_getattr_delegation(self):
        """Test that __getattr__ delegates to legacy Portia."""
        with patch("portia.client._LegacyPortia") as mock_portia_class:
            mock_portia = Mock()
            mock_portia.some_attribute = "test_value"
            mock_portia_class.return_value = mock_portia

            client = PortiaClient()

            # Should delegate to legacy Portia
            assert client.some_attribute == "test_value"

    def test_has_all_expected_methods(self):
        """Test that PortiaClient has all expected public methods."""
        client = PortiaClient()

        expected_methods = [
            "run", "arun", "plan", "aplan", "run_plan", "arun_plan",
            "resume", "aresume", "run_builder_plan", "resume_builder_plan"
        ]

        for method in expected_methods:
            assert hasattr(client, method)
            assert callable(getattr(client, method))