"""Tests for PortiaContext class."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

from portia.config import Config, StorageClass
from portia.context import PortiaContext
from portia.execution_hooks import ExecutionHooks
from portia.storage import DiskFileStorage, InMemoryStorage, PortiaCloudStorage
from portia.telemetry.telemetry_service import ProductTelemetry
from portia.tool import Tool
from portia.tool_registry import DefaultToolRegistry, ToolRegistry


class TestPortiaContext:
    """Test PortiaContext initialization and dependency management."""

    def test_init_with_defaults(self):
        """Test PortiaContext initialization with default values."""
        context = PortiaContext()

        assert context.config is not None
        assert isinstance(context.config, Config)
        assert isinstance(context.execution_hooks, ExecutionHooks)
        assert isinstance(context.telemetry, ProductTelemetry)
        assert isinstance(context.tool_registry, DefaultToolRegistry)
        assert isinstance(context.storage, InMemoryStorage)  # Default storage is memory

    def test_init_with_custom_config(self):
        """Test PortiaContext initialization with custom config."""
        custom_config = Config(
            storage_class=StorageClass.DISK,
            storage_dir="/tmp/test",
        )
        context = PortiaContext(config=custom_config)

        assert context.config is custom_config
        assert context.config.storage_class == StorageClass.DISK
        assert context.config.storage_dir == "/tmp/test"

    def test_init_with_tool_registry(self):
        """Test PortiaContext initialization with tool registry."""
        mock_tool_registry = MagicMock(spec=ToolRegistry)
        context = PortiaContext(tools=mock_tool_registry)

        assert context.tool_registry is mock_tool_registry

    def test_init_with_tool_list(self):
        """Test PortiaContext initialization with list of tools."""
        mock_tool = MagicMock(spec=Tool)
        mock_tool.id = "test_tool"
        tools = [mock_tool]
        context = PortiaContext(tools=tools)

        assert isinstance(context.tool_registry, ToolRegistry)

    def test_init_with_execution_hooks(self):
        """Test PortiaContext initialization with execution hooks."""
        custom_hooks = ExecutionHooks()
        context = PortiaContext(execution_hooks=custom_hooks)

        assert context.execution_hooks is custom_hooks

    def test_init_with_telemetry(self):
        """Test PortiaContext initialization with custom telemetry."""
        mock_telemetry = MagicMock()
        context = PortiaContext(telemetry=mock_telemetry)

        assert context.telemetry is mock_telemetry

    def test_storage_memory_initialization(self):
        """Test that memory storage is initialized correctly."""
        config = Config(storage_class=StorageClass.MEMORY)
        context = PortiaContext(config=config)

        assert isinstance(context.storage, InMemoryStorage)

    def test_storage_disk_initialization(self):
        """Test that disk storage is initialized correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(
                storage_class=StorageClass.DISK,
                storage_dir=temp_dir,
            )
            context = PortiaContext(config=config)

            assert isinstance(context.storage, DiskFileStorage)

    def test_storage_cloud_initialization(self):
        """Test that cloud storage is initialized correctly."""
        config = Config(
            storage_class=StorageClass.CLOUD,
            portia_api_key="test-key",
        )
        context = PortiaContext(config=config)

        assert isinstance(context.storage, PortiaCloudStorage)

    def test_has_portia_api_key_property(self):
        """Test has_portia_api_key property."""
        # Test without API key
        config = Config()
        context = PortiaContext(config=config)
        assert not context.has_portia_api_key

        # Test with API key
        config_with_key = Config(portia_api_key="test-key")
        context_with_key = PortiaContext(config=config_with_key)
        assert context_with_key.has_portia_api_key

    def test_all_dependencies_initialized(self):
        """Test that all required dependencies are initialized."""
        context = PortiaContext()

        # Verify all required attributes exist and are not None
        assert context.config is not None
        assert context.tool_registry is not None
        assert context.execution_hooks is not None
        assert context.telemetry is not None
        assert context.storage is not None
