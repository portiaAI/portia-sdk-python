"""Tests for Portia backward compatibility after introducing PortiaContext."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

from portia.config import Config, StorageClass
from portia.execution_hooks import ExecutionHooks
from portia.portia import Portia
from portia.storage import DiskFileStorage, InMemoryStorage, PortiaCloudStorage
from portia.telemetry.telemetry_service import ProductTelemetry
from portia.tool import Tool
from portia.tool_registry import DefaultToolRegistry, ToolRegistry


class TestPortiaBackwardCompatibility:
    """Test that Portia class maintains exact same behavior after introducing PortiaContext."""

    def test_init_with_defaults_maintains_same_attributes(self):
        """Test that Portia initialization with defaults creates same attributes as before."""
        with patch("portia.portia.logger_manager") as mock_logger_manager:
            portia = Portia()

            # Verify all expected attributes are present and accessible
            assert hasattr(portia, "config")
            assert hasattr(portia, "tool_registry")
            assert hasattr(portia, "execution_hooks")
            assert hasattr(portia, "telemetry")
            assert hasattr(portia, "storage")

            # Verify types are correct
            assert isinstance(portia.config, Config)
            assert isinstance(portia.tool_registry, DefaultToolRegistry)
            assert isinstance(portia.execution_hooks, ExecutionHooks)
            assert isinstance(portia.telemetry, ProductTelemetry)
            assert isinstance(portia.storage, InMemoryStorage)

            # Verify internal context exists but doesn't break the API
            assert hasattr(portia, "_context")

    def test_init_with_custom_config_maintains_compatibility(self):
        """Test that custom config initialization works as before."""
        custom_config = Config(
            storage_class=StorageClass.DISK,
            storage_dir="/tmp/test",
        )

        with patch("portia.portia.logger_manager") as mock_logger_manager:
            portia = Portia(config=custom_config)

            assert portia.config is custom_config
            assert portia.config.storage_class == StorageClass.DISK
            assert portia.config.storage_dir == "/tmp/test"
            assert isinstance(portia.storage, DiskFileStorage)

    def test_init_with_tool_registry_maintains_compatibility(self):
        """Test that tool registry initialization works as before."""
        mock_tool_registry = MagicMock(spec=ToolRegistry)

        with patch("portia.portia.logger_manager") as mock_logger_manager:
            portia = Portia(tools=mock_tool_registry)

            assert portia.tool_registry is mock_tool_registry

    def test_init_with_tool_list_maintains_compatibility(self):
        """Test that tool list initialization works as before."""
        mock_tool = MagicMock(spec=Tool)
        mock_tool.id = "test_tool"
        tools = [mock_tool]

        with patch("portia.portia.logger_manager") as mock_logger_manager:
            portia = Portia(tools=tools)

            assert isinstance(portia.tool_registry, ToolRegistry)

    def test_init_with_execution_hooks_maintains_compatibility(self):
        """Test that execution hooks initialization works as before."""
        custom_hooks = ExecutionHooks()

        with patch("portia.portia.logger_manager") as mock_logger_manager:
            portia = Portia(execution_hooks=custom_hooks)

            assert portia.execution_hooks is custom_hooks

    def test_init_with_telemetry_maintains_compatibility(self):
        """Test that telemetry initialization works as before."""
        mock_telemetry = MagicMock()

        with patch("portia.portia.logger_manager") as mock_logger_manager:
            portia = Portia(telemetry=mock_telemetry)

            assert portia.telemetry is mock_telemetry

    @patch("portia.portia.logger")
    def test_logging_behavior_unchanged(self, mock_logger):
        """Test that logging behavior is unchanged."""
        with patch("portia.portia.logger_manager") as mock_logger_manager, \
             patch("portia.portia.get_version", return_value="1.0.0"):

            config = Config(portia_api_key="test-key", portia_api_endpoint="https://api.portia.ai")
            portia = Portia(config=config)

            # Verify logger was configured and called
            mock_logger_manager.configure_from_config.assert_called_once_with(config)
            mock_logger.return_value.info.assert_called()

    def test_storage_initialization_unchanged(self):
        """Test that storage initialization behavior is unchanged for all storage types."""
        with patch("portia.portia.logger_manager") as mock_logger_manager:
            # Test Memory storage
            config_memory = Config(storage_class=StorageClass.MEMORY)
            portia_memory = Portia(config=config_memory)
            assert isinstance(portia_memory.storage, InMemoryStorage)

            # Test Disk storage
            with tempfile.TemporaryDirectory() as temp_dir:
                config_disk = Config(storage_class=StorageClass.DISK, storage_dir=temp_dir)
                portia_disk = Portia(config=config_disk)
                assert isinstance(portia_disk.storage, DiskFileStorage)

            # Test Cloud storage
            config_cloud = Config(storage_class=StorageClass.CLOUD, portia_api_key="test-key")
            portia_cloud = Portia(config=config_cloud)
            assert isinstance(portia_cloud.storage, PortiaCloudStorage)

    @patch("portia.portia.logger")
    def test_api_key_warning_behavior_unchanged(self, mock_logger):
        """Test that API key warning behavior is unchanged."""
        with patch("portia.portia.logger_manager") as mock_logger_manager:
            # Without API key - should warn
            portia = Portia()
            mock_logger.return_value.warning.assert_called_with(
                "No Portia API key found, Portia cloud tools and storage will not be available.",
            )

            mock_logger.reset_mock()

            # With API key - should not warn
            config_with_key = Config(portia_api_key="test-key")
            portia_with_key = Portia(config=config_with_key)
            mock_logger.return_value.warning.assert_not_called()

    def test_all_public_methods_still_exist(self):
        """Test that all expected public methods still exist."""
        with patch("portia.portia.logger_manager") as mock_logger_manager:
            portia = Portia()

            # Test existence of key public methods
            assert hasattr(portia, "run")
            assert hasattr(portia, "arun")
            assert hasattr(portia, "plan")
            assert hasattr(portia, "aplan")
            assert hasattr(portia, "initialize_end_user")
            assert hasattr(portia, "ainitialize_end_user")

            # Verify methods are callable
            assert callable(portia.run)
            assert callable(portia.arun)
            assert callable(portia.plan)
            assert callable(portia.aplan)

    def test_context_internal_implementation_detail(self):
        """Test that _context is properly internal and doesn't affect public API."""
        with patch("portia.portia.logger_manager") as mock_logger_manager:
            portia = Portia()

            # Context should exist but be internal
            assert hasattr(portia, "_context")

            # All public attributes should delegate to context properly
            assert portia.config is portia._context.config
            assert portia.tool_registry is portia._context.tool_registry
            assert portia.execution_hooks is portia._context.execution_hooks
            assert portia.telemetry is portia._context.telemetry
            assert portia.storage is portia._context.storage
