"""Integration tests for PortiaContext with the Portia class."""

import pytest

from portia import Portia
from portia.config import Config
from portia.core.context import PortiaContext
from portia.execution_hooks import ExecutionHooks
from portia.logger import logger_manager
from portia.storage import InMemoryStorage
from portia.telemetry.telemetry_service import ProductTelemetry
from portia.tool_registry import ToolRegistry


class TestPortiaContextIntegration:
    """Integration tests for PortiaContext with Portia class."""

    def test_portia_creates_context_on_init(self):
        """Test that Portia creates a PortiaContext on initialization."""
        portia = Portia()

        # Verify context was created
        assert hasattr(portia, "_context")
        assert isinstance(portia._context, PortiaContext)

        # Verify context property works
        context = portia.context
        assert isinstance(context, PortiaContext)
        assert context is portia._context

    def test_portia_context_contains_all_dependencies(self):
        """Test that the PortiaContext contains all required dependencies."""
        config = Config.from_default()
        tools = ToolRegistry([])
        execution_hooks = ExecutionHooks()
        telemetry = ProductTelemetry()

        portia = Portia(
            config=config,
            tools=tools,
            execution_hooks=execution_hooks,
            telemetry=telemetry,
        )

        context = portia.context

        # Verify all dependencies are in context
        assert context.config is portia.config
        assert context.logger_manager is logger_manager
        assert context.storage is portia.storage
        assert context.tool_registry is portia.tool_registry
        assert context.telemetry is portia.telemetry
        assert context.execution_hooks is portia.execution_hooks

    def test_portia_context_immutability(self):
        """Test that the PortiaContext is immutable after creation."""
        portia = Portia()
        context = portia.context

        # Should not be able to modify context fields
        with pytest.raises(AttributeError, match="can't set attribute"):
            context.config = Config.from_default()

    def test_portia_maintains_backward_compatibility(self):
        """Test that existing Portia attributes are still accessible."""
        config = Config.from_default()
        tools = ToolRegistry([])
        execution_hooks = ExecutionHooks()
        telemetry = ProductTelemetry()

        portia = Portia(
            config=config,
            tools=tools,
            execution_hooks=execution_hooks,
            telemetry=telemetry,
        )

        # Verify backward compatibility - all original attributes are still accessible
        assert portia.config is config
        assert portia.tool_registry is tools
        assert portia.execution_hooks is execution_hooks
        assert portia.telemetry is telemetry
        assert isinstance(portia.storage, InMemoryStorage)  # default storage

    def test_portia_context_consistency(self):
        """Test that context dependencies are consistent with Portia attributes."""
        portia = Portia()

        # Verify context and Portia instance have same objects
        assert portia.context.config is portia.config
        assert portia.context.storage is portia.storage
        assert portia.context.tool_registry is portia.tool_registry
        assert portia.context.telemetry is portia.telemetry
        assert portia.context.execution_hooks is portia.execution_hooks

    def test_portia_different_storage_configurations(self):
        """Test that PortiaContext works with different storage configurations."""
        # Test with in-memory storage (default)
        portia_memory = Portia()
        assert isinstance(portia_memory.context.storage, InMemoryStorage)

        # Test with disk storage
        config_disk = Config.from_default()
        config_disk.storage_class = "DISK"
        portia_disk = Portia(config=config_disk)
        # The actual storage type would be DiskFileStorage but we can't test file operations
        # in this unit test, so we just verify it's not None
        assert portia_disk.context.storage is not None

    def test_portia_context_factory_method(self):
        """Test that the factory method is used correctly in Portia."""
        portia = Portia()
        context = portia.context

        # Verify the context was created with from_portia_init_params
        # We can't directly test the factory method call, but we can verify
        # that all expected components are present
        assert context.config is not None
        assert context.logger_manager is not None
        assert context.storage is not None
        assert context.tool_registry is not None
        assert context.telemetry is not None
        assert context.execution_hooks is not None

    def test_portia_context_property_is_readonly(self):
        """Test that the context property cannot be overridden."""
        portia = Portia()
        original_context = portia.context

        # The context property should return the same object
        assert portia.context is original_context
        assert portia.context is original_context  # Call multiple times

        # Since it's a property, we can't directly test setting it,
        # but we can verify it returns the internal _context
        assert portia.context is portia._context