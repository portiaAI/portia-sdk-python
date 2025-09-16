"""Tests for the deprecation system and feature flags."""

import os
import warnings
from unittest.mock import Mock, patch

from portia.config import FEATURE_FLAG_PLAN_V2_DEFAULT, Config
from portia.deprecation import (
    DeprecationLogger,
    get_deprecation_logger,
    get_plan_v2_default_flag,
    set_deprecation_logger,
    warn_planv1_usage,
)
from portia.plan import Plan, PlanBuilder, PlanContext


class TestDeprecationUtilities:
    """Test deprecation utility functions."""

    def test_get_plan_v2_default_flag_false_by_default(self) -> None:
        """Test that the flag returns False by default."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_plan_v2_default_flag() is False

    def test_get_plan_v2_default_flag_true_values(self) -> None:
        """Test that the flag returns True for various true values."""
        for true_value in ("true", "True", "TRUE", "1", "yes", "Yes", "YES"):
            with patch.dict(os.environ, {"PLAN_V2_DEFAULT": true_value}):
                assert get_plan_v2_default_flag() is True, f"Failed for value: {true_value}"

    def test_get_plan_v2_default_flag_false_values(self) -> None:
        """Test that the flag returns False for various false values."""
        for false_value in ("false", "False", "FALSE", "0", "no", "No", "NO", "invalid"):
            with patch.dict(os.environ, {"PLAN_V2_DEFAULT": false_value}):
                assert get_plan_v2_default_flag() is False, f"Failed for value: {false_value}"


class TestDeprecationLogger:
    """Test DeprecationLogger class."""

    def test_init_default(self) -> None:
        """Test DeprecationLogger initialization with defaults."""
        logger = DeprecationLogger()
        assert logger.logger is None
        assert logger.use_warnings is True
        assert logger.plan_v2_default is False  # default without env var

    def test_init_with_custom_logger(self) -> None:
        """Test DeprecationLogger initialization with custom logger."""
        mock_logger = Mock()
        logger = DeprecationLogger(logger=mock_logger, use_warnings=False, plan_v2_default=True)
        assert logger.logger is mock_logger
        assert logger.use_warnings is False
        assert logger.plan_v2_default is True

    def test_warn_planv1_usage_with_logger(self) -> None:
        """Test deprecation warning with custom logger."""
        mock_logger = Mock()
        logger = DeprecationLogger(logger=mock_logger, use_warnings=False)

        logger.warn_planv1_usage("TestComponent", "TestV2", "Additional info")

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        assert "DEPRECATION: TestComponent is deprecated" in call_args
        assert "TestV2" in call_args
        assert "Additional info" in call_args

    def test_warn_planv1_usage_with_warnings(self) -> None:
        """Test deprecation warning with Python warnings."""
        logger = DeprecationLogger(use_warnings=True)

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            logger.warn_planv1_usage("TestComponent", "TestV2")

        assert len(warning_list) == 1
        assert issubclass(warning_list[0].category, DeprecationWarning)
        assert "TestComponent is deprecated" in str(warning_list[0].message)
        assert "TestV2" in str(warning_list[0].message)

    def test_warn_planv1_usage_with_feature_flag(self) -> None:
        """Test deprecation warning includes feature flag context."""
        logger = DeprecationLogger(plan_v2_default=True)

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            logger.warn_planv1_usage("TestComponent", "TestV2")

        assert len(warning_list) == 1
        message = str(warning_list[0].message)
        assert "PLAN_V2_DEFAULT is enabled" in message

    def test_warn_import_usage(self) -> None:
        """Test import-specific deprecation warning."""
        mock_logger = Mock()
        logger = DeprecationLogger(logger=mock_logger, use_warnings=False)

        logger.warn_import_usage("old_module", "OldClass", "new_module.NewClass", "Migration guide")

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        assert "DEPRECATION: OldClass is deprecated" in call_args
        assert "new_module.NewClass" in call_args
        assert "Importing OldClass from old_module" in call_args
        assert "Migration guide" in call_args


class TestGlobalDeprecationLogger:
    """Test global deprecation logger management."""

    def test_get_deprecation_logger_singleton(self) -> None:
        """Test that get_deprecation_logger returns the same instance."""
        logger1 = get_deprecation_logger()
        logger2 = get_deprecation_logger()
        assert logger1 is logger2

    def test_set_deprecation_logger(self) -> None:
        """Test setting a custom global deprecation logger."""
        custom_logger = DeprecationLogger(use_warnings=False)
        set_deprecation_logger(custom_logger)

        retrieved_logger = get_deprecation_logger()
        assert retrieved_logger is custom_logger

        # Reset for other tests
        set_deprecation_logger(DeprecationLogger())

    def test_warn_planv1_usage_global_function(self) -> None:
        """Test the global warn_planv1_usage function."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            warn_planv1_usage("GlobalTest", "GlobalTestV2", "Global context")

        assert len(warning_list) == 1
        message = str(warning_list[0].message)
        assert "GlobalTest is deprecated" in message
        assert "GlobalTestV2" in message
        assert "Global context" in message


class TestConfigFeatureFlag:
    """Test feature flag integration with Config."""

    def test_config_includes_plan_v2_default_flag(self) -> None:
        """Test that Config includes the PLAN_V2_DEFAULT feature flag."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}):
            config = Config()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True

    def test_config_plan_v2_default_false_by_default(self) -> None:
        """Test that PLAN_V2_DEFAULT is False by default in Config."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is False

    def test_config_plan_v2_default_can_be_overridden(self) -> None:
        """Test that feature flag can be overridden in Config constructor."""
        config = Config(feature_flags={FEATURE_FLAG_PLAN_V2_DEFAULT: True})
        assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True


class TestPlanV1DeprecationWarnings:
    """Test deprecation warnings for PlanV1 components."""

    def test_plan_builder_deprecation_warning(self) -> None:
        """Test that PlanBuilder emits deprecation warning on initialization."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            PlanBuilder("test query")

        # Should have deprecation warning from our custom warning and possibly from @deprecated
        deprecation_warnings = [
            w for w in warning_list if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1

        # Check our custom warning
        custom_warnings = [
            w
            for w in deprecation_warnings
            if "PlanBuilder is deprecated" in str(w.message) and "PlanBuilderV2" in str(w.message)
        ]
        assert len(custom_warnings) == 1

    def test_plan_deprecation_warning(self) -> None:
        """Test that Plan emits deprecation warning on instantiation."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            Plan(plan_context=PlanContext(query="test", tool_ids=[]), steps=[])

        # Should have deprecation warning from our custom warning
        deprecation_warnings = [
            w for w in warning_list if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1

        # Check our custom warning
        custom_warnings = [
            w
            for w in deprecation_warnings
            if "Plan is deprecated" in str(w.message) and "PlanV2" in str(w.message)
        ]
        assert len(custom_warnings) == 1

    def test_plan_builder_build_method_warning(self) -> None:
        """Test that building a plan through PlanBuilder triggers warnings."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            builder = PlanBuilder("test query")
            builder.step("test step", output="output1")
            plan = builder.build()

        # Should have warnings from both PlanBuilder init and Plan instantiation
        deprecation_warnings = [
            w for w in warning_list if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 2

        # Check that we have warnings for both components
        planbuilder_warnings = [
            w for w in deprecation_warnings if "PlanBuilder is deprecated" in str(w.message)
        ]
        plan_warnings = [
            w
            for w in deprecation_warnings
            if "Plan is deprecated" in str(w.message) and "PlanV2" in str(w.message)
        ]

        assert len(planbuilder_warnings) >= 1
        assert len(plan_warnings) >= 1
        assert isinstance(plan, Plan)


class TestEnvironmentVariableIntegration:
    """Test integration with environment variables."""

    def test_deprecation_with_env_var_enabled(self) -> None:
        """Test deprecation warnings when PLAN_V2_DEFAULT is enabled."""
        with (
            patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}),
            warnings.catch_warnings(record=True) as warning_list,
        ):
            warnings.simplefilter("always")
            PlanBuilder("test")

        # Find our custom warning
        custom_warnings = [
            w
            for w in warning_list
            if (
                issubclass(w.category, DeprecationWarning)
                and "PLAN_V2_DEFAULT is enabled" in str(w.message)
            )
        ]
        assert len(custom_warnings) == 1

    def test_deprecation_without_env_var(self) -> None:
        """Test deprecation warnings when PLAN_V2_DEFAULT is not enabled."""
        with (
            patch.dict(os.environ, {}, clear=True),
            warnings.catch_warnings(record=True) as warning_list,
        ):
            warnings.simplefilter("always")
            PlanBuilder("test")

        # Should not mention feature flag in warnings
        for w in warning_list:
            if issubclass(w.category, DeprecationWarning):
                assert "PLAN_V2_DEFAULT is enabled" not in str(w.message)


class TestStackLevels:
    """Test that warnings point to the correct location in user code."""

    def test_plan_builder_warning_stacklevel(self) -> None:
        """Test that PlanBuilder warning points to user code, not internal code."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            def user_function() -> PlanBuilder:
                return PlanBuilder("test")

            user_function()

        custom_warnings = [
            w
            for w in warning_list
            if (
                issubclass(w.category, DeprecationWarning)
                and "PlanBuilder is deprecated" in str(w.message)
            )
        ]
        assert len(custom_warnings) == 1

        # The warning should point to user_function, not to internal deprecation code
        warning = custom_warnings[0]
        assert warning.filename.endswith("test_deprecation.py")
        # The line should be in user_function (the return statement)
        # We can't test exact line numbers as they may change, but we can test the function name
