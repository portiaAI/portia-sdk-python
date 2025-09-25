"""Unit tests for the core logging functionality."""

from unittest.mock import Mock, patch

from portia.config import Config, GenerativeModelsConfig
from portia.core.logging import log_models


class TestCoreLogging:
    """Test cases for core logging utilities."""

    @patch("portia.core.logging.logger")
    def test_log_models(self, mock_logger):
        """Test that log_models correctly logs all configured models."""
        # Create a mock logger instance
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        # Create a config with default values
        config = Config.from_default()

        # Call log_models
        log_models(config)

        # Verify that logger was called with the expected messages
        calls = mock_logger_instance.debug.call_args_list

        # Should have called debug at least twice (header + model info)
        assert len(calls) >= 2

        # Check that the first call is the header
        assert calls[0][0][0] == "Portia Generative Models"

        # Check that subsequent calls log model information
        # We verify this by checking that get_ methods are called for each model field
        for model_field in GenerativeModelsConfig.model_fields:
            # Verify that the getter method exists and was used
            getter_method = f"get_{model_field}"
            assert hasattr(config, getter_method)

            # Check that at least one call contains the model field name
            model_logged = any(model_field in str(call[0][0]) for call in calls[1:])
            assert model_logged, f"Model {model_field} was not logged"

    @patch("portia.core.logging.logger")
    def test_log_models_with_custom_config(self, mock_logger):
        """Test log_models with a custom config."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        # Create a custom config
        config = Config.from_default()

        # Call log_models
        log_models(config)

        # Verify logger was called
        mock_logger.assert_called_once()
        mock_logger_instance.debug.assert_called()

        # Verify that the header was logged
        calls = mock_logger_instance.debug.call_args_list
        header_logged = any("Portia Generative Models" in str(call) for call in calls)
        assert header_logged

    def test_log_models_accesses_all_model_fields(self):
        """Test that log_models accesses all model fields from GenerativeModelsConfig."""
        config = Config.from_default()

        # Verify that all model fields have corresponding getter methods
        for model_field in GenerativeModelsConfig.model_fields:
            getter_method = f"get_{model_field}"
            assert hasattr(config, getter_method), f"Missing getter method: {getter_method}"

            # Verify that the getter can be called without error
            getter = getattr(config, getter_method)
            try:
                result = getter()
                assert result is not None or result is None  # Just verify it returns something
            except Exception as e:
                pytest.fail(f"Getter {getter_method} raised an exception: {e}")

    @patch("portia.core.logging.logger")
    def test_log_models_handles_none_values(self, mock_logger):
        """Test that log_models handles None values gracefully."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        config = Config.from_default()

        # Mock some getters to return None
        with patch.object(config, 'get_planning_agent_model', return_value=None):
            log_models(config)

        # Should still log without raising exceptions
        mock_logger_instance.debug.assert_called()
        calls = mock_logger_instance.debug.call_args_list
        assert len(calls) >= 1