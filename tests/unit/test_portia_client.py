"""Tests for PortiaClient facade."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from portia.client import PortiaClient
from portia.config import Config
from portia.context import PortiaContext
from portia.execution_hooks import ExecutionHooks
from portia.tool import Tool
from portia.tool_registry import ToolRegistry


class TestPortiaClient:
    """Test PortiaClient facade initialization and API."""

    def test_init_with_defaults(self):
        """Test PortiaClient initialization with default values."""
        client = PortiaClient()

        assert isinstance(client.context, PortiaContext)
        assert client.context.config is not None
        assert client.context.execution_hooks is not None
        assert client.context.telemetry is not None
        assert client.context.tool_registry is not None
        assert client.context.storage is not None

    def test_init_with_custom_parameters(self):
        """Test PortiaClient initialization with custom parameters."""
        custom_config = Config()
        custom_hooks = ExecutionHooks()
        mock_telemetry = MagicMock()
        mock_tool_registry = MagicMock(spec=ToolRegistry)

        client = PortiaClient(
            config=custom_config,
            tools=mock_tool_registry,
            execution_hooks=custom_hooks,
            telemetry=mock_telemetry,
        )

        assert client.context.config is custom_config
        assert client.context.execution_hooks is custom_hooks
        assert client.context.telemetry is mock_telemetry
        assert client.context.tool_registry is mock_tool_registry

    def test_context_delegation(self):
        """Test that PortiaClient properly delegates to PortiaContext."""
        client = PortiaClient()

        # Verify that context is properly initialized
        assert hasattr(client, "context")
        assert isinstance(client.context, PortiaContext)

    def test_run_method_placeholder(self):
        """Test that run method has proper placeholder."""
        client = PortiaClient()

        with pytest.raises(NotImplementedError, match="PortiaClient.run will be implemented"):
            client.run("test query")

    def test_arun_method_placeholder(self):
        """Test that arun method has proper placeholder."""
        client = PortiaClient()

        with pytest.raises(NotImplementedError, match="PortiaClient.arun will be implemented"):
            import asyncio
            asyncio.run(client.arun("test query"))

    def test_plan_method_placeholder(self):
        """Test that plan method has proper placeholder."""
        client = PortiaClient()

        with pytest.raises(NotImplementedError, match="PortiaClient.plan will be implemented"):
            client.plan("test query")

    def test_aplan_method_placeholder(self):
        """Test that aplan method has proper placeholder."""
        client = PortiaClient()

        with pytest.raises(NotImplementedError, match="PortiaClient.aplan will be implemented"):
            import asyncio
            asyncio.run(client.aplan("test query"))

    def test_init_with_tool_list(self):
        """Test PortiaClient initialization with list of tools."""
        mock_tool = MagicMock(spec=Tool)
        mock_tool.id = "test_tool"
        tools = [mock_tool]

        client = PortiaClient(tools=tools)

        # Should create a ToolRegistry from the list
        assert isinstance(client.context.tool_registry, ToolRegistry)

    def test_maintains_same_api_signature_as_portia(self):
        """Test that PortiaClient methods maintain the same signature as Portia."""
        client = PortiaClient()

        # Test that methods exist with expected signatures
        assert hasattr(client, "run")
        assert hasattr(client, "arun")
        assert hasattr(client, "plan")
        assert hasattr(client, "aplan")

        # Check method signatures match expected parameters
        import inspect

        run_sig = inspect.signature(client.run)
        expected_params = ["query", "tools", "example_plans", "end_user",
                          "plan_run_inputs", "structured_output_schema", "use_cached_plan"]
        assert list(run_sig.parameters.keys())[1:] == expected_params  # Skip 'self'

        arun_sig = inspect.signature(client.arun)
        assert list(arun_sig.parameters.keys())[1:] == expected_params  # Skip 'self'

        plan_sig = inspect.signature(client.plan)
        assert list(plan_sig.parameters.keys())[1:] == expected_params  # Skip 'self'

        aplan_sig = inspect.signature(client.aplan)
        assert list(aplan_sig.parameters.keys())[1:] == expected_params  # Skip 'self'
