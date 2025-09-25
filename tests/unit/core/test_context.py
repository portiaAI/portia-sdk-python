"""Unit tests for the PortiaContext dataclass."""

import pytest
from unittest.mock import Mock

from portia.config import Config
from portia.core.context import PortiaContext
from portia.execution_hooks import ExecutionHooks
from portia.logger import LoggerManager
from portia.storage import InMemoryStorage
from portia.telemetry.telemetry_service import BaseProductTelemetry
from portia.tool_registry import ToolRegistry


class TestPortiaContext:
    """Test cases for PortiaContext."""

    def test_context_creation(self):
        """Test creating a PortiaContext with all required components."""
        # Create mock components
        config = Config.from_default()
        logger_manager = LoggerManager()
        storage = InMemoryStorage()
        tool_registry = ToolRegistry([])
        telemetry = Mock(spec=BaseProductTelemetry)
        execution_hooks = ExecutionHooks()

        # Create context
        context = PortiaContext(
            config=config,
            logger_manager=logger_manager,
            storage=storage,
            tool_registry=tool_registry,
            telemetry=telemetry,
            execution_hooks=execution_hooks,
        )

        # Verify all components are set correctly
        assert context.config is config
        assert context.logger_manager is logger_manager
        assert context.storage is storage
        assert context.tool_registry is tool_registry
        assert context.telemetry is telemetry
        assert context.execution_hooks is execution_hooks

    def test_context_is_dataclass(self):
        """Test that PortiaContext is properly defined as a dataclass."""
        # Create context
        config = Config.from_default()
        logger_manager = LoggerManager()
        storage = InMemoryStorage()
        tool_registry = ToolRegistry([])
        telemetry = Mock(spec=BaseProductTelemetry)
        execution_hooks = ExecutionHooks()

        context = PortiaContext(
            config=config,
            logger_manager=logger_manager,
            storage=storage,
            tool_registry=tool_registry,
            telemetry=telemetry,
            execution_hooks=execution_hooks,
        )

        # Verify dataclass behavior
        assert hasattr(context, "__dataclass_fields__")
        assert len(context.__dataclass_fields__) == 6

        # Verify field names
        field_names = set(context.__dataclass_fields__.keys())
        expected_fields = {
            "config",
            "logger_manager",
            "storage",
            "tool_registry",
            "telemetry",
            "execution_hooks",
        }
        assert field_names == expected_fields

    def test_context_attributes_accessible(self):
        """Test that all context attributes are accessible."""
        config = Config.from_default()
        logger_manager = LoggerManager()
        storage = InMemoryStorage()
        tool_registry = ToolRegistry([])
        telemetry = Mock(spec=BaseProductTelemetry)
        execution_hooks = ExecutionHooks()

        context = PortiaContext(
            config=config,
            logger_manager=logger_manager,
            storage=storage,
            tool_registry=tool_registry,
            telemetry=telemetry,
            execution_hooks=execution_hooks,
        )

        # Test attribute access doesn't raise exceptions
        assert context.config is not None
        assert context.logger_manager is not None
        assert context.storage is not None
        assert context.tool_registry is not None
        assert context.telemetry is not None
        assert context.execution_hooks is not None