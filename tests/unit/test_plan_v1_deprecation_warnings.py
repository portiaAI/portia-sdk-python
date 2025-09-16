"""Tests for PlanV1 import-time and initialization deprecation warnings."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from portia.deprecation import deprecation_logger


class TestPlanBuilderDeprecationWarnings:
    """Test deprecation warnings for PlanBuilder (V1)."""

    def setUp(self) -> None:
        """Reset deprecation warnings before each test."""
        deprecation_logger.reset_warnings()

    def test_plan_builder_init_triggers_warning(self) -> None:
        """Test that initializing PlanBuilder triggers a deprecation warning."""
        self.setUp()

        with warnings.catch_warnings(record=True) as w, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            warnings.simplefilter("always")
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            # Import and initialize PlanBuilder
            from portia.plan import PlanBuilder

            builder = PlanBuilder("test query")

            # Check that warnings were triggered
            assert len(w) >= 1

            # Find our specific deprecation warning
            plan_builder_warnings = [
                warning for warning in w
                if "PlanBuilder" in str(warning.message) and
                issubclass(warning.category, DeprecationWarning)
            ]

            assert len(plan_builder_warnings) >= 1
            warning = plan_builder_warnings[0]
            assert "PlanBuilder is deprecated" in str(warning.message)
            assert "PlanBuilderV2" in str(warning.message)

            # Check that Portia logger was called
            mock_portia_logger.warning.assert_called()
            warning_calls = [
                call for call in mock_portia_logger.warning.call_args_list
                if "DEPRECATED: PlanBuilder" in str(call[0][0])
            ]
            assert len(warning_calls) >= 1

    def test_plan_builder_init_warning_only_once_per_session(self) -> None:
        """Test that PlanBuilder initialization warning is only shown once per session."""
        self.setUp()

        with warnings.catch_warnings(record=True) as w, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            warnings.simplefilter("always")
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            from portia.plan import PlanBuilder

            # First initialization should trigger warning
            builder1 = PlanBuilder("test query 1")
            initial_warning_count = len([
                warning for warning in w
                if "PlanBuilder" in str(warning.message) and
                issubclass(warning.category, DeprecationWarning)
            ])

            # Second initialization should not trigger additional warning
            builder2 = PlanBuilder("test query 2")
            final_warning_count = len([
                warning for warning in w
                if "PlanBuilder" in str(warning.message) and
                issubclass(warning.category, DeprecationWarning)
            ])

            assert initial_warning_count == final_warning_count

    def test_plan_builder_import_does_not_trigger_warning(self) -> None:
        """Test that importing PlanBuilder alone doesn't trigger warnings."""
        self.setUp()

        with warnings.catch_warnings(record=True) as w, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            warnings.simplefilter("always")
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            # Just importing shouldn't trigger deprecation warning
            from portia.plan import PlanBuilder

            # Check that no deprecation warnings were triggered just from import
            plan_builder_warnings = [
                warning for warning in w
                if "PlanBuilder" in str(warning.message) and
                issubclass(warning.category, DeprecationWarning)
            ]

            # Shouldn't have warnings just from import (only from instantiation)
            assert len(plan_builder_warnings) == 0

    def test_plan_builder_warning_includes_migration_guide(self) -> None:
        """Test that PlanBuilder warning includes migration guide URL."""
        self.setUp()

        with warnings.catch_warnings(record=True) as w, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            warnings.simplefilter("always")
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            from portia.plan import PlanBuilder

            builder = PlanBuilder("test query")

            # Find the deprecation warning
            plan_builder_warnings = [
                warning for warning in w
                if "PlanBuilder" in str(warning.message) and
                issubclass(warning.category, DeprecationWarning)
            ]

            assert len(plan_builder_warnings) >= 1
            warning_message = str(plan_builder_warnings[0].message)

            # Check that migration guide is mentioned
            assert "migration guide" in warning_message.lower()
            assert "docs.portialabs.ai" in warning_message

    def test_plan_builder_warning_with_filtering(self) -> None:
        """Test PlanBuilder warning behavior with different warning filters."""
        self.setUp()

        # Test with warnings ignored
        with warnings.catch_warnings(record=True) as w, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            warnings.simplefilter("ignore", DeprecationWarning)
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            from portia.plan import PlanBuilder

            builder = PlanBuilder("test query")

            # Python warnings should be filtered out
            plan_builder_warnings = [
                warning for warning in w
                if "PlanBuilder" in str(warning.message) and
                issubclass(warning.category, DeprecationWarning)
            ]
            assert len(plan_builder_warnings) == 0

            # But Portia logger should still be called
            mock_portia_logger.warning.assert_called()

    def test_plan_builder_with_structured_output_schema(self) -> None:
        """Test that PlanBuilder with structured output schema still triggers warning."""
        self.setUp()

        with warnings.catch_warnings(record=True) as w, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            warnings.simplefilter("always")
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            from portia.plan import PlanBuilder
            from pydantic import BaseModel

            class TestSchema(BaseModel):
                result: str

            builder = PlanBuilder("test query", structured_output_schema=TestSchema)

            # Should still trigger deprecation warning
            plan_builder_warnings = [
                warning for warning in w
                if "PlanBuilder" in str(warning.message) and
                issubclass(warning.category, DeprecationWarning)
            ]

            assert len(plan_builder_warnings) >= 1

    def test_plan_builder_warning_stacklevel(self) -> None:
        """Test that the warning has the correct stack level."""
        self.setUp()

        with patch("portia.deprecation.warnings.warn") as mock_warn, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            from portia.plan import PlanBuilder

            builder = PlanBuilder("test query")

            # Check that warnings.warn was called with correct stacklevel
            mock_warn.assert_called()
            call_args = mock_warn.call_args

            # Should have stacklevel parameter
            assert "stacklevel" in call_args.kwargs
            # Stacklevel should be reasonable (typically 2-4 for this case)
            assert 1 <= call_args.kwargs["stacklevel"] <= 5


class TestPlanBuilderIntegration:
    """Test integration aspects of PlanBuilder deprecation warnings."""

    def test_plan_builder_functionality_still_works(self) -> None:
        """Test that PlanBuilder still works despite deprecation warnings."""
        deprecation_logger.reset_warnings()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for this test

            from portia.plan import PlanBuilder

            # PlanBuilder should still work normally
            builder = PlanBuilder("test query")

            # Basic functionality should work
            assert builder.query == "test query"
            assert isinstance(builder.steps, list)
            assert len(builder.steps) == 0
            assert isinstance(builder.plan_inputs, list)
            assert len(builder.plan_inputs) == 0

            # Should be able to add steps
            builder.step("Test task", output="test_output")
            assert len(builder.steps) == 1
            assert builder.steps[0].task == "Test task"

    def test_plan_builder_build_still_works(self) -> None:
        """Test that building a plan still works despite deprecation warnings."""
        deprecation_logger.reset_warnings()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for this test

            from portia.plan import PlanBuilder

            builder = PlanBuilder("test query")
            builder.step("Test task", output="test_output")

            # Should be able to build the plan
            plan = builder.build()

            assert plan is not None
            assert plan.plan_context.query == "test query"
            assert len(plan.steps) == 1

    def test_multiple_plan_builders_independent_warnings(self) -> None:
        """Test that warnings work correctly with multiple PlanBuilder instances."""
        deprecation_logger.reset_warnings()

        with warnings.catch_warnings(record=True) as w, patch(
            "portia.deprecation.logger"
        ) as mock_logger_func:
            warnings.simplefilter("always")
            mock_portia_logger = MagicMock()
            mock_logger_func.return_value = mock_portia_logger

            from portia.plan import PlanBuilder

            # Create multiple instances
            builders = []
            for i in range(3):
                builders.append(PlanBuilder(f"query {i}"))

            # Should get warning for each instance (because of once_per_session)
            # Actually, it should only warn once per session, not per instance
            plan_builder_warnings = [
                warning for warning in w
                if "PlanBuilder" in str(warning.message) and
                issubclass(warning.category, DeprecationWarning)
            ]

            # Should only have one warning due to once_per_session behavior
            assert len(plan_builder_warnings) == 1

    def test_decorator_warning_still_present(self) -> None:
        """Test that the @deprecated decorator warning is still present."""
        from portia.plan import PlanBuilder

        # The class should still have the @deprecated decorator
        # This is more of a static check, but we can verify it exists
        assert hasattr(PlanBuilder, "__deprecated__") or "deprecated" in str(PlanBuilder.__doc__ or "")


class TestPlanV1ImportWarnings:
    """Test import-time warnings for PlanV1 components."""

    def test_importing_from_portia_init_includes_deprecation_module(self) -> None:
        """Test that importing from portia.__init__ includes deprecation handling."""
        # This test verifies that the deprecation module is imported
        # when importing from the main portia package

        # Reset warnings
        deprecation_logger.reset_warnings()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            # This import should include the deprecation module
            import portia

            # Verify deprecation module is available
            assert hasattr(portia, 'deprecation') or hasattr(portia, '_')

            # The deprecation_logger should be accessible
            from portia.deprecation import deprecation_logger as dl
            assert dl is not None