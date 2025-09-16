"""Tests for the deprecation utilities."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from portia.deprecation import (
    DeprecationLogger,
    check_and_warn_import,
    deprecation_logger,
    warn_deprecated,
    warn_plan_v1_usage,
)


class TestDeprecationLogger:
    """Test the DeprecationLogger class."""

    def test_warn_basic(self) -> None:
        """Test basic warning functionality."""
        logger = DeprecationLogger()
        logger.reset_warnings()

        with patch("portia.deprecation.warnings.warn") as mock_warn, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            logger.warn("TestItem", "TestItem is deprecated")

            mock_warn.assert_called_once_with(
                "TestItem is deprecated", category=DeprecationWarning, stacklevel=2
            )
            mock_portia_logger.warning.assert_called_once_with(
                "DEPRECATED: TestItem - TestItem is deprecated"
            )

    def test_warn_once_per_session(self) -> None:
        """Test that warnings are only shown once per session by default."""
        logger = DeprecationLogger()
        logger.reset_warnings()

        with patch("portia.deprecation.warnings.warn") as mock_warn, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            # First warning should be shown
            logger.warn("TestItem", "TestItem is deprecated")
            assert mock_warn.call_count == 1
            assert mock_portia_logger.warning.call_count == 1

            # Second warning for the same item should be ignored
            logger.warn("TestItem", "TestItem is deprecated")
            assert mock_warn.call_count == 1
            assert mock_portia_logger.warning.call_count == 1

            # Different item should still trigger warning
            logger.warn("OtherItem", "OtherItem is deprecated")
            assert mock_warn.call_count == 2
            assert mock_portia_logger.warning.call_count == 2

    def test_warn_multiple_times(self) -> None:
        """Test that warnings can be shown multiple times when once_per_session=False."""
        logger = DeprecationLogger()
        logger.reset_warnings()

        with patch("portia.deprecation.warnings.warn") as mock_warn, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            # First warning
            logger.warn("TestItem", "TestItem is deprecated", once_per_session=False)
            assert mock_warn.call_count == 1
            assert mock_portia_logger.warning.call_count == 1

            # Second warning for the same item should still be shown
            logger.warn("TestItem", "TestItem is deprecated", once_per_session=False)
            assert mock_warn.call_count == 2
            assert mock_portia_logger.warning.call_count == 2

    def test_warn_with_migration(self) -> None:
        """Test warning with migration guidance."""
        logger = DeprecationLogger()
        logger.reset_warnings()

        with patch("portia.deprecation.warnings.warn") as mock_warn, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            logger.warn_with_migration(
                "OldClass", "NewClass", additional_info="See docs for details"
            )

            expected_message = (
                "OldClass is deprecated and will be removed in a future version. "
                "Use NewClass instead. See docs for details"
            )
            mock_warn.assert_called_once_with(
                expected_message, category=DeprecationWarning, stacklevel=2
            )
            mock_portia_logger.warning.assert_called_once_with(
                f"DEPRECATED: OldClass - {expected_message}"
            )

    def test_warn_plan_v1_usage(self) -> None:
        """Test PlanV1 specific warnings."""
        logger = DeprecationLogger()
        logger.reset_warnings()

        with patch("portia.deprecation.warnings.warn") as mock_warn, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            logger.warn_plan_v1_usage("PlanBuilder")

            expected_message = (
                "PlanBuilder is deprecated and will be removed in a future version. "
                "Use PlanBuilderV2 instead. "
                "See the migration guide at https://docs.portialabs.ai/plan-v2-migration"
            )
            mock_warn.assert_called_once_with(
                expected_message, category=DeprecationWarning, stacklevel=2
            )
            mock_portia_logger.warning.assert_called_once_with(
                f"DEPRECATED: PlanBuilder - {expected_message}"
            )

    def test_warn_plan_v1_unknown_item(self) -> None:
        """Test PlanV1 warning for unknown items uses generic replacement."""
        logger = DeprecationLogger()
        logger.reset_warnings()

        with patch("portia.deprecation.warnings.warn") as mock_warn, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            logger.warn_plan_v1_usage("UnknownItem")

            expected_message = (
                "UnknownItem is deprecated and will be removed in a future version. "
                "Use the V2 equivalent instead. "
                "See the migration guide at https://docs.portialabs.ai/plan-v2-migration"
            )
            mock_warn.assert_called_once()
            args, kwargs = mock_warn.call_args
            assert expected_message in args[0]

    def test_reset_warnings(self) -> None:
        """Test that reset_warnings allows warnings to be shown again."""
        logger = DeprecationLogger()
        logger.reset_warnings()

        with patch("portia.deprecation.warnings.warn") as mock_warn, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            # First warning
            logger.warn("TestItem", "TestItem is deprecated")
            assert mock_warn.call_count == 1

            # Second warning should be ignored
            logger.warn("TestItem", "TestItem is deprecated")
            assert mock_warn.call_count == 1

            # Reset and try again
            logger.reset_warnings()
            logger.warn("TestItem", "TestItem is deprecated")
            assert mock_warn.call_count == 2

    def test_custom_warning_category(self) -> None:
        """Test using a custom warning category."""
        logger = DeprecationLogger()
        logger.reset_warnings()

        with patch("portia.deprecation.warnings.warn") as mock_warn, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            logger.warn("TestItem", "TestItem is deprecated", category=FutureWarning)

            mock_warn.assert_called_once_with(
                "TestItem is deprecated", category=FutureWarning, stacklevel=2
            )

    def test_custom_stack_level(self) -> None:
        """Test using a custom stack level."""
        logger = DeprecationLogger()
        logger.reset_warnings()

        with patch("portia.deprecation.warnings.warn") as mock_warn, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            logger.warn("TestItem", "TestItem is deprecated", stacklevel=5)

            mock_warn.assert_called_once_with(
                "TestItem is deprecated", category=DeprecationWarning, stacklevel=5
            )


class TestConvenienceFunctions:
    """Test the convenience functions."""

    def test_warn_deprecated(self) -> None:
        """Test the warn_deprecated convenience function."""
        with patch("portia.deprecation.deprecation_logger") as mock_logger:
            warn_deprecated("TestItem", "TestItem is deprecated")

            mock_logger.warn.assert_called_once_with(
                deprecated_item="TestItem",
                message="TestItem is deprecated",
                category=DeprecationWarning,
                stacklevel=3,  # Adjusted for extra function call
                once_per_session=True,
            )

    def test_warn_plan_v1_usage_function(self) -> None:
        """Test the warn_plan_v1_usage convenience function."""
        with patch("portia.deprecation.deprecation_logger") as mock_logger:
            warn_plan_v1_usage("PlanBuilder")

            mock_logger.warn_plan_v1_usage.assert_called_once_with(
                deprecated_item="PlanBuilder",
                stacklevel=3,  # Adjusted for extra function call
                once_per_session=True,
            )

    def test_check_and_warn_import(self) -> None:
        """Test the check_and_warn_import function."""
        deprecated_items = {"OldClass": "NewClass", "AnotherOldClass": "AnotherNewClass"}
        import_globals = {"OldClass": object(), "SomeOtherItem": object()}

        with patch("portia.deprecation.deprecation_logger") as mock_logger:
            check_and_warn_import("test_module", deprecated_items, import_globals)

            mock_logger.warn_with_migration.assert_called_once_with(
                deprecated_item="test_module.OldClass",
                replacement="test_module.NewClass",
                stacklevel=3,
            )


class TestGlobalDeprecationLogger:
    """Test the global deprecation logger instance."""

    def test_global_logger_exists(self) -> None:
        """Test that the global deprecation logger exists."""
        assert isinstance(deprecation_logger, DeprecationLogger)

    def test_global_logger_functionality(self) -> None:
        """Test that the global logger works as expected."""
        deprecation_logger.reset_warnings()

        with patch("portia.deprecation.warnings.warn") as mock_warn, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            deprecation_logger.warn("GlobalTest", "This is a test warning")

            mock_warn.assert_called_once()
            mock_portia_logger.warning.assert_called_once()


class TestIntegrationWithWarningsSystem:
    """Test integration with Python's warnings system."""

    def test_warnings_filter_integration(self) -> None:
        """Test that deprecation warnings work with Python's warnings filter."""
        logger = DeprecationLogger()
        logger.reset_warnings()

        # Test with warnings filtered
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Catch all warnings

            with patch("portia.deprecation.logger") as mock_logger_func:
                mock_portia_logger = MagicMock()
                mock_logger_func.return_value = mock_portia_logger

                logger.warn("TestItem", "TestItem is deprecated")

                # Check that warning was recorded by warnings system
                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "TestItem is deprecated" in str(w[0].message)

                # Check that Portia logger was also called
                mock_portia_logger.warning.assert_called_once()

    def test_warnings_ignored_when_filtered(self) -> None:
        """Test that warnings can be filtered but Portia logger still gets called."""
        logger = DeprecationLogger()
        logger.reset_warnings()

        # Test with warnings filtered out
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", DeprecationWarning)

            with patch("portia.deprecation.logger") as mock_logger_func:
                mock_portia_logger = MagicMock()
                mock_logger_func.return_value = mock_portia_logger

                logger.warn("TestItem", "TestItem is deprecated")

                # Warning should be filtered out by warnings system
                assert len(w) == 0

                # But Portia logger should still be called
                mock_portia_logger.warning.assert_called_once()