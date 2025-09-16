"""Tests for the PLAN_V2_DEFAULT feature flag functionality."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from portia.config import FEATURE_FLAG_PLAN_V2_DEFAULT, Config


class TestPlanV2FeatureFlag:
    """Test the PLAN_V2_DEFAULT feature flag."""

    def test_plan_v2_default_false_by_default(self) -> None:
        """Test that PLAN_V2_DEFAULT is False by default."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is False

    def test_plan_v2_default_true_from_env_true(self) -> None:
        """Test that PLAN_V2_DEFAULT=true enables the flag."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True

    def test_plan_v2_default_true_from_env_1(self) -> None:
        """Test that PLAN_V2_DEFAULT=1 enables the flag."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "1"}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True

    def test_plan_v2_default_true_from_env_yes(self) -> None:
        """Test that PLAN_V2_DEFAULT=yes enables the flag."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "yes"}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True

    def test_plan_v2_default_true_from_env_on(self) -> None:
        """Test that PLAN_V2_DEFAULT=on enables the flag."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "on"}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True

    def test_plan_v2_default_true_case_insensitive(self) -> None:
        """Test that the environment variable is case insensitive."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "TRUE"}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True

        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "True"}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True

        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "YES"}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True

    def test_plan_v2_default_false_from_env_false(self) -> None:
        """Test that PLAN_V2_DEFAULT=false disables the flag."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "false"}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is False

    def test_plan_v2_default_false_from_env_0(self) -> None:
        """Test that PLAN_V2_DEFAULT=0 disables the flag."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "0"}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is False

    def test_plan_v2_default_false_from_env_no(self) -> None:
        """Test that PLAN_V2_DEFAULT=no disables the flag."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "no"}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is False

    def test_plan_v2_default_false_from_env_off(self) -> None:
        """Test that PLAN_V2_DEFAULT=off disables the flag."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "off"}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is False

    def test_plan_v2_default_false_from_invalid_env(self) -> None:
        """Test that invalid environment values default to False."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "maybe"}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is False

        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "invalid"}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is False

        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": ""}, clear=True):
            config = Config.from_default()
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is False

    def test_plan_v2_default_override_with_feature_flags(self) -> None:
        """Test that feature_flags parameter can override environment variable."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}, clear=True):
            # Environment says True, but we override to False
            config = Config.from_default(
                feature_flags={FEATURE_FLAG_PLAN_V2_DEFAULT: False}
            )
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is False

        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "false"}, clear=True):
            # Environment says False, but we override to True
            config = Config.from_default(
                feature_flags={FEATURE_FLAG_PLAN_V2_DEFAULT: True}
            )
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True

    def test_plan_v2_default_mixed_feature_flags(self) -> None:
        """Test that mixing environment and explicit feature flags works correctly."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}, clear=True):
            config = Config.from_default(
                feature_flags={
                    "other_flag": True,
                    # PLAN_V2_DEFAULT not specified, should use env value
                }
            )
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True
            assert config.feature_flags["other_flag"] is True

    def test_plan_v2_feature_flag_constant_accessible(self) -> None:
        """Test that the feature flag constant is accessible from portia.__init__."""
        # This test verifies that the constant can be imported
        from portia import FEATURE_FLAG_PLAN_V2_DEFAULT as imported_constant

        assert imported_constant == "plan_v2_default"

    def test_feature_flag_documentation_present(self) -> None:
        """Test that the feature flag is documented in the Config class."""
        config = Config.from_default()

        # Check that the field has proper documentation
        feature_flags_field = Config.model_fields["feature_flags"]
        assert "plan_v2_default" in feature_flags_field.description
        assert "PlanBuilderV2" in feature_flags_field.description
        assert "PLAN_V2_DEFAULT" in feature_flags_field.description

    def test_plan_v2_default_preserves_other_feature_flags(self) -> None:
        """Test that setting PLAN_V2_DEFAULT doesn't affect other feature flags."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}, clear=True):
            config = Config.from_default()

            # The agent memory flag should still be present and True
            from portia.config import FEATURE_FLAG_AGENT_MEMORY_ENABLED

            assert config.feature_flags[FEATURE_FLAG_AGENT_MEMORY_ENABLED] is True
            assert config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] is True

    def test_config_model_validation_with_feature_flag(self) -> None:
        """Test that the Config model validates correctly with the feature flag."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}, clear=True):
            # This should not raise any validation errors
            try:
                config = Config.from_default()
                assert isinstance(config.feature_flags, dict)
                assert FEATURE_FLAG_PLAN_V2_DEFAULT in config.feature_flags
                assert isinstance(config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT], bool)
            except Exception as e:
                pytest.fail(f"Config validation failed with feature flag: {e}")


class TestPlanV2FeatureFlagBehavior:
    """Test behavioral aspects of the PLAN_V2_DEFAULT feature flag."""

    def test_feature_flag_can_be_used_for_conditional_logic(self) -> None:
        """Test that the feature flag can be used for conditional logic."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}, clear=True):
            config = Config.from_default()

            # Simulate code that would use the feature flag
            if config.feature_flags.get(FEATURE_FLAG_PLAN_V2_DEFAULT, False):
                use_plan_v2 = True
            else:
                use_plan_v2 = False

            assert use_plan_v2 is True

        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "false"}, clear=True):
            config = Config.from_default()

            if config.feature_flags.get(FEATURE_FLAG_PLAN_V2_DEFAULT, False):
                use_plan_v2 = True
            else:
                use_plan_v2 = False

            assert use_plan_v2 is False

    def test_feature_flag_persistence_across_config_instances(self) -> None:
        """Test that the feature flag is consistent across different config instances."""
        with patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}, clear=True):
            config1 = Config.from_default()
            config2 = Config.from_default()

            assert (
                config1.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT]
                == config2.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT]
            )

            both_enabled = (
                config1.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT] and
                config2.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT]
            )
            assert both_enabled is True