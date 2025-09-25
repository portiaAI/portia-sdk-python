"""Tests for PortiaContext."""

from unittest.mock import Mock

import pytest

from portia.config import Config
from portia.core.context import PortiaContext
from portia.execution_hooks import ExecutionHooks
from portia.logger import logger_manager
from portia.storage import InMemoryStorage
from portia.telemetry.telemetry_service import ProductTelemetry
from portia.tool_registry import ToolRegistry


class TestPortiaContext:
    """Test cases for PortiaContext class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing."""
        return {
            "config": Mock(spec=Config),
            "logger_manager": Mock(spec=logger_manager.__class__),
            "storage": Mock(spec=InMemoryStorage),
            "tool_registry": Mock(spec=ToolRegistry),
            "telemetry": Mock(spec=ProductTelemetry),
            "execution_hooks": Mock(spec=ExecutionHooks),
        }

    def test_portia_context_creation(self, mock_dependencies):
        """Test that PortiaContext can be created with all dependencies."""
        context = PortiaContext(**mock_dependencies)

        assert context.config is mock_dependencies["config"]
        assert context.logger_manager is mock_dependencies["logger_manager"]
        assert context.storage is mock_dependencies["storage"]
        assert context.tool_registry is mock_dependencies["tool_registry"]
        assert context.telemetry is mock_dependencies["telemetry"]
        assert context.execution_hooks is mock_dependencies["execution_hooks"]

    def test_portia_context_immutable(self, mock_dependencies):
        """Test that PortiaContext is immutable (frozen dataclass)."""
        context = PortiaContext(**mock_dependencies)

        with pytest.raises(AttributeError, match="can't set attribute"):
            context.config = Mock()

    def test_from_portia_init_params_factory(self, mock_dependencies):
        """Test the factory method creates PortiaContext correctly."""
        context = PortiaContext.from_portia_init_params(**mock_dependencies)

        assert context.config is mock_dependencies["config"]
        assert context.logger_manager is mock_dependencies["logger_manager"]
        assert context.storage is mock_dependencies["storage"]
        assert context.tool_registry is mock_dependencies["tool_registry"]
        assert context.telemetry is mock_dependencies["telemetry"]
        assert context.execution_hooks is mock_dependencies["execution_hooks"]

    def test_portia_context_integration_with_real_objects(self):
        """Test PortiaContext with real dependency objects."""
        config = Config.from_default()
        storage = InMemoryStorage()
        tool_registry = ToolRegistry([])
        telemetry = ProductTelemetry()
        execution_hooks = ExecutionHooks()

        context = PortiaContext.from_portia_init_params(
            config=config,
            logger_manager=logger_manager,
            storage=storage,
            tool_registry=tool_registry,
            telemetry=telemetry,
            execution_hooks=execution_hooks,
        )

        # Verify all objects are properly stored
        assert context.config is config
        assert context.logger_manager is logger_manager
        assert context.storage is storage
        assert context.tool_registry is tool_registry
        assert context.telemetry is telemetry
        assert context.execution_hooks is execution_hooks

    def test_portia_context_type_annotations(self, mock_dependencies):
        """Test that PortiaContext maintains proper type annotations."""
        context = PortiaContext(**mock_dependencies)

        # Check that the fields have the expected annotations
        annotations = PortiaContext.__annotations__
        assert "config" in annotations
        assert "logger_manager" in annotations
        assert "storage" in annotations
        assert "tool_registry" in annotations
        assert "telemetry" in annotations
        assert "execution_hooks" in annotations