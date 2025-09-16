"""Integration tests for deprecation warnings and feature flags."""

import os
import warnings
from unittest.mock import patch


class TestDeprecationIntegration:
    """Integration tests for deprecation functionality."""

    def test_end_to_end_plan_v1_deprecation_flow(self):
        """Test complete deprecation flow for PlanV1 classes."""
        # Test 1: Feature flag functionality from environment
        from portia.deprecation import is_plan_v2_default

        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}):
            assert is_plan_v2_default() is True

        with patch.dict(os.environ, {}, clear=True):
            assert is_plan_v2_default() is False

        # Test 2: PlanBuilder deprecation warning
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            from portia.plan import PlanBuilder
            builder = PlanBuilder(query="test query")

            # Verify deprecation warnings were emitted
            deprecation_warnings = [
                w for w in warning_list
                if issubclass(w.category, DeprecationWarning) and
                "PlanBuilder" in str(w.message)
            ]
            assert len(deprecation_warnings) >= 1

        # Test 3: Plan deprecation warning
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            from portia.plan import Plan, PlanContext
            context = PlanContext(query="test", tool_ids=[])
            plan = Plan(plan_context=context, steps=[])

            # Verify deprecation warnings were emitted
            deprecation_warnings = [
                w for w in warning_list
                if issubclass(w.category, DeprecationWarning) and
                "Plan" in str(w.message)
            ]
            assert len(deprecation_warnings) >= 1

        # Test 4: Functionality still works despite warnings
        assert builder.query == "test query"
        assert plan.plan_context.query == "test"

    def test_feature_flag_environment_variable_integration(self):
        """Test that feature flag integrates properly with environment variables."""
        from portia.deprecation import is_plan_v2_default

        # Test default behavior (no env var)
        with patch.dict(os.environ, {}, clear=True):
            assert is_plan_v2_default() is False

        # Test with env var set
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}):
            assert is_plan_v2_default() is True

        # Test with env var set to false
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "false"}):
            assert is_plan_v2_default() is False

    def test_deprecation_warnings_can_be_silenced(self):
        """Test that deprecation warnings can be properly silenced."""
        # Silence deprecation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            from portia.plan import Plan, PlanBuilder, PlanContext
            builder = PlanBuilder(query="test")
            context = PlanContext(query="test", tool_ids=[])
            plan = Plan(plan_context=context, steps=[])

            # Should not raise any warnings and functionality should work
            assert builder.query == "test"
            assert plan.plan_context.query == "test"

    def test_migration_path_works(self):
        """Test that migration path from V1 to V2 works correctly."""
        # Import both V1 and V2 classes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from portia.builder.plan_builder_v2 import PlanBuilderV2  # V2 (new)
            from portia.plan import PlanBuilder  # V1 (deprecated)

        # Create builders with both versions
        v1_builder = PlanBuilder(query="test v1")
        v2_builder = PlanBuilderV2(label="test v2")

        # Both should work
        assert v1_builder.query == "test v1"
        assert v2_builder.plan.label == "test v2"

        # V1 can build a plan
        v1_plan = v1_builder.step("Test step", output="$output_0").build()
        assert len(v1_plan.steps) == 1

        # V2 can build a plan
        v2_plan = v2_builder.llm_step(task="Test step").build()
        assert len(v2_plan.steps) == 1


class TestDocumentationAndUsage:
    """Test that deprecation warnings provide useful guidance."""

    def test_deprecation_messages_are_informative(self):
        """Test that deprecation warnings provide clear migration guidance."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            from portia.plan import PlanBuilder

            builder = PlanBuilder(query="test")

            # Find deprecation warnings
            deprecation_warnings = [
                str(w.message) for w in warning_list
                if issubclass(w.category, DeprecationWarning)
            ]

            # Should have at least one informative message
            assert any(
                "PlanBuilder" in msg and
                ("PlanBuilderV2" in msg or "deprecated" in msg)
                for msg in deprecation_warnings
            )

    def test_feature_flag_behavior_documented(self):
        """Test that feature flag behavior is consistent and predictable."""
        from portia.deprecation import PLAN_V2_DEFAULT_ENV, is_plan_v2_default

        # Environment variable name should be consistent
        assert PLAN_V2_DEFAULT_ENV == "PLAN_V2_DEFAULT"

        # Function behavior should be predictable
        test_cases = [
            ("true", True),
            ("TRUE", True),
            ("True", True),
            ("false", False),
            ("FALSE", False),
            ("", False),
            ("invalid", False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"PLAN_V2_DEFAULT": env_value}):
                assert is_plan_v2_default() == expected, f"Failed for env_value='{env_value}'"
