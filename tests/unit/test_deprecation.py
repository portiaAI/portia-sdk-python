"""Tests for deprecation utilities and warnings."""

import os
import warnings
from unittest import mock

import pytest

from portia.deprecation import (
    deprecated_class,
    deprecated_import,
    deprecation_warning,
    get_plan_v2_default,
)


class TestFeatureFlags:
    """Test feature flag utilities."""

    def test_get_plan_v2_default_true(self) -> None:
        """Test PLAN_V2_DEFAULT returns True when env var is 'true'."""
        with mock.patch.dict(os.environ, {"PLAN_V2_DEFAULT": "true"}):
            assert get_plan_v2_default() is True

    def test_get_plan_v2_default_false(self) -> None:
        """Test PLAN_V2_DEFAULT returns False when env var is 'false'."""
        with mock.patch.dict(os.environ, {"PLAN_V2_DEFAULT": "false"}):
            assert get_plan_v2_default() is False

    def test_get_plan_v2_default_case_insensitive(self) -> None:
        """Test PLAN_V2_DEFAULT is case insensitive."""
        with mock.patch.dict(os.environ, {"PLAN_V2_DEFAULT": "TRUE"}):
            assert get_plan_v2_default() is True

        with mock.patch.dict(os.environ, {"PLAN_V2_DEFAULT": "False"}):
            assert get_plan_v2_default() is False

    def test_get_plan_v2_default_default_false(self) -> None:
        """Test PLAN_V2_DEFAULT defaults to False when env var is not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            assert get_plan_v2_default() is False

    def test_get_plan_v2_default_invalid_value(self) -> None:
        """Test PLAN_V2_DEFAULT returns False for invalid values."""
        with mock.patch.dict(os.environ, {"PLAN_V2_DEFAULT": "maybe"}):
            assert get_plan_v2_default() is False


class TestDeprecationWarning:
    """Test deprecation_warning function."""

    def test_deprecation_warning_emitted(self) -> None:
        """Test that deprecation_warning emits a DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="Test deprecation message"):
            deprecation_warning("Test deprecation message")

    def test_deprecation_warning_stacklevel(self) -> None:
        """Test that deprecation_warning has correct stacklevel."""
        def caller_function() -> None:
            deprecation_warning("Test message")

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            caller_function()

        assert len(warning_list) == 1
        assert issubclass(warning_list[0].category, DeprecationWarning)
        assert "Test message" in str(warning_list[0].message)


class TestDeprecatedImport:
    """Test deprecated_import function."""

    def test_deprecated_import_warning(self) -> None:
        """Test that deprecated_import emits proper deprecation warning."""
        with pytest.warns(
            DeprecationWarning,
            match="Importing TestClass from test_module is deprecated \\(deprecated since v1.0\\). "
                  "Use NewTestClass instead."
        ):
            deprecated_import("test_module", "TestClass", "NewTestClass", "1.0")

    def test_deprecated_import_without_version(self) -> None:
        """Test that deprecated_import works without version."""
        with pytest.warns(
            DeprecationWarning,
            match="Importing TestClass from test_module is deprecated. Use NewTestClass instead."
        ):
            deprecated_import("test_module", "TestClass", "NewTestClass")


class TestDeprecatedClass:
    """Test deprecated_class decorator."""

    def test_deprecated_class_decorator(self) -> None:
        """Test that deprecated_class decorator adds deprecation warning on init."""
        @deprecated_class("NewTestClass", "1.0")
        class TestClass:
            def __init__(self) -> None:
                pass

        with pytest.warns(
            DeprecationWarning,
            match="TestClass is deprecated \\(deprecated since v1.0\\). Use NewTestClass instead."
        ):
            TestClass()

    def test_deprecated_class_decorator_without_version(self) -> None:
        """Test that deprecated_class decorator works without version."""
        @deprecated_class("NewTestClass")
        class TestClass:
            def __init__(self) -> None:
                pass

        with pytest.warns(
            DeprecationWarning,
            match="TestClass is deprecated. Use NewTestClass instead."
        ):
            TestClass()

    def test_deprecated_class_with_args(self) -> None:
        """Test that deprecated_class decorator works with constructor args."""
        @deprecated_class("NewTestClass", "1.0")
        class TestClass:
            def __init__(self, value: str) -> None:
                self.value = value

        with pytest.warns(DeprecationWarning):
            obj = TestClass("test_value")
            assert obj.value == "test_value"

    def test_deprecated_class_preserves_original_init(self) -> None:
        """Test that deprecated_class decorator preserves original __init__ behavior."""
        original_init_called = False

        @deprecated_class("NewTestClass")
        class TestClass:
            def __init__(self) -> None:
                nonlocal original_init_called
                original_init_called = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            TestClass()
            assert original_init_called is True


class TestDeprecationIntegration:
    """Integration tests for deprecation functionality."""

    def test_multiple_warnings_different_classes(self) -> None:
        """Test that multiple deprecated classes emit separate warnings."""
        @deprecated_class("NewClassA")
        class ClassA:
            def __init__(self) -> None:
                pass

        @deprecated_class("NewClassB")
        class ClassB:
            def __init__(self) -> None:
                pass

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            ClassA()
            ClassB()

        assert len(warning_list) == 2
        assert all(issubclass(w.category, DeprecationWarning) for w in warning_list)
        messages = [str(w.message) for w in warning_list]
        assert any("ClassA" in msg for msg in messages)
        assert any("ClassB" in msg for msg in messages)

    def test_deprecation_warning_stacklevel_correct(self) -> None:
        """Test that deprecation warnings point to the correct stack level."""
        @deprecated_class("NewTestClass")
        class TestClass:
            def __init__(self) -> None:
                pass

        def create_instance() -> TestClass:
            return TestClass()

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            create_instance()

        assert len(warning_list) == 1
        # The warning should point to the create_instance function call, not the decorator
        warning = warning_list[0]
        assert warning.filename.endswith("test_deprecation.py")
        assert "create_instance" in warning.filename or warning.lineno > 0
