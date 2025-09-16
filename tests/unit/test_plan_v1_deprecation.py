"""Tests for Plan V1 deprecation warnings and feature flag integration."""

import os
import warnings
from unittest import mock

import pytest

from portia.config import FEATURE_FLAG_PLAN_V2_DEFAULT, Config
from portia.plan import Plan, PlanBuilder, PlanContext


class TestPlanV1DeprecationWarnings:
    """Test deprecation warnings for Plan V1 classes."""

    def test_plan_builder_deprecation_warning(self) -> None:
        """Test that PlanBuilder emits deprecation warning on initialization."""
        with pytest.warns(
            DeprecationWarning,
            match="PlanBuilder is deprecated \\(deprecated since v0.8.0\\). "
                  "Use PlanBuilderV2 instead."
        ):
            PlanBuilder("test query")

    def test_plan_deprecation_warning(self) -> None:
        """Test that Plan emits deprecation warning on initialization."""
        plan_context = PlanContext(query="test", tool_ids=[])

        with pytest.warns(
            DeprecationWarning,
            match="Plan is deprecated \\(deprecated since v0.8.0\\). Use PlanV2 instead."
        ):
            Plan(plan_context=plan_context, steps=[])

    def test_plan_builder_functionality_preserved(self) -> None:
        """Test that PlanBuilder functionality is preserved despite deprecation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            builder = PlanBuilder("test query")
            builder.step("Test step", output="$test_output")
            plan = builder.build()

            assert plan.plan_context.query == "test query"
            assert len(plan.steps) == 1
            assert plan.steps[0].task == "Test step"
            assert plan.steps[0].output == "$test_output"

    def test_plan_functionality_preserved(self) -> None:
        """Test that Plan functionality is preserved despite deprecation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            plan_context = PlanContext(query="test query", tool_ids=["tool1"])
            plan = Plan(plan_context=plan_context, steps=[])

            assert plan.plan_context.query == "test query"
            assert plan.plan_context.tool_ids == ["tool1"]
            assert len(plan.steps) == 0

    def test_import_deprecation_warnings_disabled_in_tests(self) -> None:
        """Test that import-time warnings are properly handled in test environment."""
        # This test ensures that our import-time deprecation warnings don't break tests
        # The warnings should be emitted but not cause test failures
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            # Re-import to trigger import-time warnings
            import importlib

            import portia
            importlib.reload(portia)

            # Check that deprecation warnings were emitted for Plan and PlanBuilder
            deprecation_warnings = [
                w for w in warning_list
                if issubclass(w.category, DeprecationWarning)
                and ("Plan" in str(w.message) or "PlanBuilder" in str(w.message))
            ]

            # We expect at least 2 warnings: one for Plan, one for PlanBuilder
            assert len(deprecation_warnings) >= 2


class TestPlanV2FeatureFlag:
    """Test PLAN_V2_DEFAULT feature flag integration."""

    def test_feature_flag_defaults_to_false(self) -> None:
        """Test that PLAN_V2_DEFAULT feature flag defaults to False."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is False

    def test_feature_flag_set_to_true(self) -> None:
        """Test that PLAN_V2_DEFAULT feature flag can be set to True."""
        with mock.patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True

    def test_feature_flag_set_to_false(self) -> None:
        """Test that PLAN_V2_DEFAULT feature flag can be explicitly set to False."""
        with mock.patch.dict(os.environ, {"PLAN_V2_DEFAULT": "false"}):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is False

    def test_feature_flag_case_insensitive(self) -> None:
        """Test that PLAN_V2_DEFAULT feature flag is case insensitive."""
        test_cases = [
            ("TRUE", True),
            ("True", True),
            ("true", True),
            ("FALSE", False),
            ("False", False),
            ("false", False),
        ]

        for env_value, expected in test_cases:
            with mock.patch.dict(os.environ, {"PLAN_V2_DEFAULT": env_value}):
                config = Config.from_default()
                assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is expected

    def test_feature_flag_invalid_values_default_to_false(self) -> None:
        """Test that invalid PLAN_V2_DEFAULT values default to False."""
        invalid_values = ["maybe", "1", "0", "yes", "no", ""]

        for invalid_value in invalid_values:
            with mock.patch.dict(os.environ, {"PLAN_V2_DEFAULT": invalid_value}):
                config = Config.from_default()
                assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is False

    def test_feature_flag_can_be_overridden_in_config(self) -> None:
        """Test that feature flag can be overridden in Config constructor."""
        with mock.patch.dict(os.environ, {"PLAN_V2_DEFAULT": "false"}):
            config = Config.from_default(feature_flags={FEATURE_FLAG_PLAN_V2_DEFAULT: True})
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True

    def test_feature_flag_in_existing_config(self) -> None:
        """Test that feature flag is properly included alongside existing flags."""
        with mock.patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}):
            config = Config.from_default(feature_flags={"custom_flag": True})

            # Both flags should be present
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True
            assert config.feature_flags["custom_flag"] is True
            # The default agent memory flag should also be present
            assert "feature_flag_agent_memory_enabled" in config.feature_flags


class TestBackwardsCompatibility:
    """Test that V1 classes still work for backwards compatibility."""

    def test_plan_builder_complete_workflow(self) -> None:
        """Test complete PlanBuilder workflow works with deprecation warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            # Test a complete workflow
            builder = PlanBuilder("Find the weather")
            builder.step("Get current weather", tool_id="weather_tool", output="$weather")
            builder.step("Format weather info", output="$formatted_weather")
            builder.input("location", "The location to get weather for", 0)
            builder.plan_input("api_key", "Weather API key")

            plan = builder.build()

            # Verify all components work
            assert plan.plan_context.query == "Find the weather"
            assert len(plan.steps) == 2
            assert len(plan.plan_inputs) == 1
            assert plan.steps[0].tool_id == "weather_tool"
            assert plan.steps[0].inputs[0].name == "location"
            assert plan.plan_inputs[0].name == "api_key"

    def test_plan_validation_still_works(self) -> None:
        """Test that Plan validation still works with deprecation warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            plan_context = PlanContext(query="test", tool_ids=[])

            # Test that validation still catches duplicate outputs
            from portia.plan import Step
            steps = [
                Step(task="Step 1", output="$duplicate"),
                Step(task="Step 2", output="$duplicate"),  # Same output
            ]

            with pytest.raises(ValueError, match="Outputs \\+ conditions must be unique"):
                Plan(plan_context=plan_context, steps=steps)
