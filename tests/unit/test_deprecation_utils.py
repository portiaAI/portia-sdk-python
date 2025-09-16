"""Tests for deprecation utilities."""

import os
import warnings
from unittest.mock import patch

import pytest

from portia.deprecation_utils import (
    deprecated,
    get_plan_v2_default_flag,
    log_deprecation_warning,
    warn_on_v1_import,
)
from portia.logger import logger


class TestGetPlanV2DefaultFlag:
    """Tests for the PLAN_V2_DEFAULT feature flag."""

    def test_returns_true_when_env_var_is_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_plan_v2_default_flag returns True when PLAN_V2_DEFAULT is 'true'."""
        monkeypatch.setenv("PLAN_V2_DEFAULT", "true")
        assert get_plan_v2_default_flag() is True

    def test_returns_true_when_env_var_is_true_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that get_plan_v2_default_flag handles case-insensitive 'true'."""
        for value in ["TRUE", "True", "TrUe"]:
            monkeypatch.setenv("PLAN_V2_DEFAULT", value)
            assert get_plan_v2_default_flag() is True

    def test_returns_false_when_env_var_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_plan_v2_default_flag returns False when PLAN_V2_DEFAULT is 'false'."""
        monkeypatch.setenv("PLAN_V2_DEFAULT", "false")
        assert get_plan_v2_default_flag() is False

    def test_returns_false_when_env_var_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_plan_v2_default_flag returns False when PLAN_V2_DEFAULT is empty."""
        monkeypatch.setenv("PLAN_V2_DEFAULT", "")
        assert get_plan_v2_default_flag() is False

    def test_returns_false_when_env_var_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_plan_v2_default_flag returns False when PLAN_V2_DEFAULT is not set."""
        monkeypatch.delenv("PLAN_V2_DEFAULT", raising=False)
        assert get_plan_v2_default_flag() is False

    def test_returns_false_for_any_other_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_plan_v2_default_flag returns False for any other value."""
        for value in ["yes", "1", "on", "enabled", "invalid"]:
            monkeypatch.setenv("PLAN_V2_DEFAULT", value)
            assert get_plan_v2_default_flag() is False


class TestLogDeprecationWarning:
    """Tests for log_deprecation_warning function."""

    def test_logs_deprecation_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that log_deprecation_warning logs the expected message."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            log_deprecation_warning("OldClass", "NewClass")

        # Check logger output
        assert "[DEPRECATION] OldClass is deprecated. Use NewClass instead." in caplog.text

        # Check Python warning was issued
        assert len(warning_list) == 1
        assert issubclass(warning_list[0].category, DeprecationWarning)
        assert "OldClass is deprecated. Use NewClass instead." in str(warning_list[0].message)

    def test_logs_deprecation_warning_with_version(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that log_deprecation_warning includes version info when provided."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            log_deprecation_warning("OldClass", "NewClass", version="2.0")

        expected_message = (
            "OldClass is deprecated and will be removed in version 2.0. Use NewClass instead."
        )

        # Check logger output
        assert f"[DEPRECATION] {expected_message}" in caplog.text

        # Check Python warning was issued
        assert len(warning_list) == 1
        assert expected_message in str(warning_list[0].message)

    def test_uses_custom_warning_category(self) -> None:
        """Test that log_deprecation_warning uses custom warning category."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            log_deprecation_warning("OldClass", "NewClass", category=FutureWarning)

        assert len(warning_list) == 1
        assert issubclass(warning_list[0].category, FutureWarning)


class TestDeprecatedDecorator:
    """Tests for the deprecated decorator."""

    def test_decorator_logs_warning_on_function_call(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that deprecated decorator logs warning when function is called."""
        @deprecated("new_function")
        def old_function():
            return "result"

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            result = old_function()

        # Function should still work
        assert result == "result"

        # Check logger output
        assert "[DEPRECATION] old_function is deprecated. Use new_function instead." in caplog.text

        # Check Python warning was issued
        assert len(warning_list) == 1
        assert "old_function is deprecated. Use new_function instead." in str(
            warning_list[0].message
        )

    def test_decorator_preserves_function_metadata(self) -> None:
        """Test that deprecated decorator preserves function metadata."""
        @deprecated("new_function")
        def old_function():
            """A deprecated function."""
            return "result"

        assert old_function.__name__ == "old_function"
        assert old_function.__doc__ == "A deprecated function."

    def test_decorator_works_with_arguments(self) -> None:
        """Test that deprecated decorator works with functions that have arguments."""
        @deprecated("new_function")
        def old_function(a, b, c=None):
            return a + b + (c or 0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for this test
            result = old_function(1, 2, c=3)

        assert result == 6


class TestWarnOnV1Import:
    """Tests for warn_on_v1_import function."""

    def test_warns_when_flag_is_enabled(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that warn_on_v1_import logs warning when PLAN_V2_DEFAULT is enabled."""
        monkeypatch.setenv("PLAN_V2_DEFAULT", "true")

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            warn_on_v1_import("OldModule", "NewModule")

        # Check logger output
        expected_message = (
            "Importing OldModule is deprecated. Use importing NewModule instead."
        )
        assert f"[DEPRECATION] {expected_message}" in caplog.text

        # Check Python warning was issued with FutureWarning category
        assert len(warning_list) == 1
        assert issubclass(warning_list[0].category, FutureWarning)
        assert expected_message in str(warning_list[0].message)

    def test_does_not_warn_when_flag_is_disabled(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that warn_on_v1_import does not log warning when PLAN_V2_DEFAULT is disabled."""
        monkeypatch.setenv("PLAN_V2_DEFAULT", "false")

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            warn_on_v1_import("OldModule", "NewModule")

        # Should not log or warn
        assert "[DEPRECATION]" not in caplog.text
        assert len(warning_list) == 0

    def test_does_not_warn_when_flag_is_not_set(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that warn_on_v1_import does not log warning when PLAN_V2_DEFAULT is not set."""
        monkeypatch.delenv("PLAN_V2_DEFAULT", raising=False)

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            warn_on_v1_import("OldModule", "NewModule")

        # Should not log or warn
        assert "[DEPRECATION]" not in caplog.text
        assert len(warning_list) == 0


class TestIntegrationWithActualClasses:
    """Integration tests with actual Plan and PlanBuilder classes."""

    def test_plan_builder_logs_deprecation_on_init(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that PlanBuilder logs deprecation warning on instantiation."""
        from portia.plan import PlanBuilder

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for cleaner test output
            builder = PlanBuilder("test query")

        # Should log deprecation warning
        assert "[DEPRECATION] PlanBuilder is deprecated. Use PlanBuilderV2 instead." in caplog.text

    def test_plan_logs_deprecation_on_init(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that Plan logs deprecation warning on instantiation."""
        from portia.plan import Plan, PlanContext

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for cleaner test output
            plan_context = PlanContext(query="test", tool_ids=[])
            plan = Plan(plan_context=plan_context, steps=[])

        # Should log deprecation warning
        assert "[DEPRECATION] Plan is deprecated. Use PlanV2 instead." in caplog.text

    def test_import_warnings_when_flag_enabled(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that importing plan module logs warnings when flag is enabled."""
        monkeypatch.setenv("PLAN_V2_DEFAULT", "true")

        # Clear any previous log entries
        caplog.clear()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for cleaner test output
            # This will trigger the module-level warn_on_v1_import calls
            import importlib
            import sys

            # Remove the module if it's already loaded to force re-import
            if "portia.plan" in sys.modules:
                del sys.modules["portia.plan"]

            importlib.import_module("portia.plan")

        # Should have logged import warnings
        logs = caplog.text
        assert "Importing portia.plan.PlanBuilder is deprecated" in logs
        assert "Importing portia.plan.Plan is deprecated" in logs