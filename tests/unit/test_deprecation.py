"""Tests for deprecation utilities and warnings."""

import os
import warnings
from unittest.mock import patch

from portia.deprecation import (
    deprecated_class,
    deprecated_function,
    is_plan_v2_default,
    log_deprecation_warning,
    warn_on_import,
)


class TestFeatureFlags:
    """Test feature flag functionality."""

    def test_is_plan_v2_default_false_by_default(self):
        """Test that PLAN_V2_DEFAULT is False by default."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_plan_v2_default() is False

    def test_is_plan_v2_default_true_when_set(self):
        """Test that PLAN_V2_DEFAULT is True when environment variable is set."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}):
            assert is_plan_v2_default() is True

    def test_is_plan_v2_default_case_insensitive(self):
        """Test that PLAN_V2_DEFAULT handles various case formats."""
        test_cases = ["true", "TRUE", "True", "TrUe"]
        for value in test_cases:
            with patch.dict(os.environ, {"PLAN_V2_DEFAULT": value}):
                assert is_plan_v2_default() is True

    def test_is_plan_v2_default_false_for_other_values(self):
        """Test that PLAN_V2_DEFAULT is False for non-true values."""
        test_cases = ["false", "FALSE", "0", "no", "off", ""]
        for value in test_cases:
            with patch.dict(os.environ, {"PLAN_V2_DEFAULT": value}):
                assert is_plan_v2_default() is False


class TestDeprecationLogging:
    """Test deprecation logging functionality."""

    def test_log_deprecation_warning_basic(self):
        """Test basic deprecation warning logging."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            log_deprecation_warning("OldClass")

            assert len(warning_list) == 1
            assert issubclass(warning_list[0].category, DeprecationWarning)
            assert "'OldClass' is deprecated" in str(warning_list[0].message)
            assert "will be removed in a future version" in str(warning_list[0].message)

    def test_log_deprecation_warning_with_replacement(self):
        """Test deprecation warning with replacement suggestion."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            log_deprecation_warning("OldClass", replacement="NewClass")

            assert len(warning_list) == 1
            message = str(warning_list[0].message)
            assert "'OldClass' is deprecated" in message
            assert "use 'NewClass' instead" in message

    def test_log_deprecation_warning_with_version(self):
        """Test deprecation warning with version information."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            log_deprecation_warning("OldClass", version="2.0.0")

            assert len(warning_list) == 1
            message = str(warning_list[0].message)
            assert "'OldClass' is deprecated" in message
            assert "will be removed in version 2.0.0" in message

    def test_log_deprecation_warning_with_replacement_and_version(self):
        """Test deprecation warning with both replacement and version."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            log_deprecation_warning("OldClass", replacement="NewClass", version="2.0.0")

            assert len(warning_list) == 1
            message = str(warning_list[0].message)
            assert "'OldClass' is deprecated" in message
            assert "use 'NewClass' instead" in message
            assert "will be removed in version 2.0.0" in message


class TestDeprecatedClassDecorator:
    """Test deprecated class decorator functionality."""

    def test_deprecated_class_basic(self):
        """Test basic deprecated class decorator."""

        @deprecated_class()
        class OldClass:
            def __init__(self):
                self.value = 42

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            instance = OldClass()

            assert len(warning_list) == 1
            assert issubclass(warning_list[0].category, DeprecationWarning)
            assert "'OldClass' is deprecated" in str(warning_list[0].message)
            assert instance.value == 42  # Ensure class still works

    def test_deprecated_class_with_replacement(self):
        """Test deprecated class decorator with replacement."""

        @deprecated_class(replacement="NewClass")
        class OldClass:
            pass

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            OldClass()

            assert len(warning_list) == 1
            message = str(warning_list[0].message)
            assert "'OldClass' is deprecated" in message
            assert "use 'NewClass' instead" in message

    def test_deprecated_class_with_args(self):
        """Test deprecated class decorator with constructor arguments."""

        @deprecated_class(replacement="NewClass")
        class OldClass:
            def __init__(self, value: int):
                self.value = value

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            instance = OldClass(123)

            assert len(warning_list) == 1
            assert instance.value == 123


class TestDeprecatedFunctionDecorator:
    """Test deprecated function decorator functionality."""

    def test_deprecated_function_basic(self):
        """Test basic deprecated function decorator."""

        @deprecated_function()
        def old_function():
            return "result"

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            result = old_function()

            assert len(warning_list) == 1
            assert issubclass(warning_list[0].category, DeprecationWarning)
            assert "'old_function' is deprecated" in str(warning_list[0].message)
            assert result == "result"  # Ensure function still works

    def test_deprecated_function_with_replacement(self):
        """Test deprecated function decorator with replacement."""

        @deprecated_function(replacement="new_function")
        def old_function():
            return "result"

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            old_function()

            assert len(warning_list) == 1
            message = str(warning_list[0].message)
            assert "'old_function' is deprecated" in message
            assert "use 'new_function' instead" in message

    def test_deprecated_function_with_args(self):
        """Test deprecated function decorator with arguments."""

        @deprecated_function(replacement="new_function")
        def old_function(x: int, y: int = 10) -> int:
            return x + y

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            result = old_function(5, y=15)

            assert len(warning_list) == 1
            assert result == 20


class TestWarnOnImport:
    """Test import-time warning functionality."""

    def test_warn_on_import_basic(self):
        """Test basic import-time warnings."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            warn_on_import("mymodule", ["OldClass"])

            assert len(warning_list) == 1
            assert issubclass(warning_list[0].category, DeprecationWarning)
            assert "'mymodule.OldClass' is deprecated" in str(warning_list[0].message)

    def test_warn_on_import_multiple_items(self):
        """Test import-time warnings for multiple items."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            warn_on_import("mymodule", ["OldClass", "OldFunction"])

            assert len(warning_list) == 2
            messages = [str(w.message) for w in warning_list]
            assert any("'mymodule.OldClass' is deprecated" in msg for msg in messages)
            assert any("'mymodule.OldFunction' is deprecated" in msg for msg in messages)

    def test_warn_on_import_with_replacement_module(self):
        """Test import-time warnings with replacement module."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            warn_on_import("oldmodule", ["OldClass"], "newmodule")

            assert len(warning_list) == 1
            message = str(warning_list[0].message)
            assert "'oldmodule.OldClass' is deprecated" in message
            assert "use 'newmodule.OldClass' instead" in message


class TestIntegrationWithConfig:
    """Test integration with configuration system."""

    def test_feature_flag_in_config(self):
        """Test that feature flag is properly integrated with config system."""
        from portia.config import FEATURE_FLAG_PLAN_V2_DEFAULT, Config

        # Test with environment variable not set
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is False

        # Test with environment variable set to true
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}):
            config = Config()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True

    def test_feature_flag_override_in_config(self):
        """Test that feature flag can be overridden in config."""
        from portia.config import FEATURE_FLAG_PLAN_V2_DEFAULT, Config

        # Override the environment setting with explicit config
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "false"}):
            config = Config(feature_flags={FEATURE_FLAG_PLAN_V2_DEFAULT: True})
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True


class TestDeprecatedPlanClasses:
    """Test deprecation warnings for actual Plan and PlanBuilder classes."""

    def test_plan_builder_deprecation_warning(self):
        """Test that PlanBuilder emits deprecation warning."""
        from portia.plan import PlanBuilder

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            builder = PlanBuilder(query="test")

            # Should have warnings from both @deprecated and @deprecated_class decorators
            assert len(warning_list) >= 1
            has_deprecation_warning = any(
                issubclass(w.category, DeprecationWarning) and
                "PlanBuilder" in str(w.message) and
                "deprecated" in str(w.message).lower()
                for w in warning_list
            )
            assert has_deprecation_warning
            assert builder.query == "test"  # Ensure functionality still works

    def test_plan_class_deprecation_warning(self):
        """Test that Plan emits deprecation warning."""
        from portia.plan import Plan, PlanContext

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            context = PlanContext(query="test", tool_ids=[])
            plan = Plan(plan_context=context, steps=[])

            # Should have deprecation warning from @deprecated_class decorator
            assert len(warning_list) >= 1
            has_deprecation_warning = any(
                issubclass(w.category, DeprecationWarning) and
                "Plan" in str(w.message) and
                "deprecated" in str(w.message).lower()
                for w in warning_list
            )
            assert has_deprecation_warning
            assert plan.plan_context.query == "test"  # Ensure functionality still works

    def test_import_warnings_triggered(self):
        """Test that import-time warnings are triggered."""
        # This test is tricky because imports happen at module load time
        # We can at least verify that the warn_on_import function is called
        # in the __init__.py, but the actual warnings would have been
        # emitted when the test suite imported portia modules

        # We can test that the function works as expected
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            warn_on_import("portia", ["Plan", "PlanBuilder"], "portia.builder")

            assert len(warning_list) == 2
            messages = [str(w.message) for w in warning_list]
            assert any("'portia.Plan' is deprecated" in msg for msg in messages)
            assert any("'portia.PlanBuilder' is deprecated" in msg for msg in messages)
