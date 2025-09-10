"""Tests for common classes."""

import json
from unittest import mock
from uuid import UUID

import pytest
from pydantic import BaseModel, Field

from portia.common import (
    EXTRAS_GROUPS_DEPENDENCIES,
    PortiaEnum,
    combine_args_kwargs,
    singleton,
    validate_extras_dependencies,
)
from portia.prefixed_uuid import PrefixedUUID


def test_portia_enum() -> None:
    """Test PortiaEnums can enumerate."""

    class MyEnum(PortiaEnum):
        OK = "OK"

    assert MyEnum.enumerate() == (("OK", "OK"),)


def test_combine_args_kwargs() -> None:
    """Test combining args and kwargs into a single dictionary."""

    class Comment(BaseModel):
        """Test comment model."""

        body: str
        public: bool

    comment = Comment(body="The issue is being reviewed.", public=True)
    result = combine_args_kwargs(
        98765,  # ticket_id as positional arg
        comment,  # comment as positional arg
        subject="Updated ticket subject",
        priority="high",
        status="in_progress",
        assignee_id=12345,
    )

    assert result == {
        "0": 98765,
        "1": {"body": "The issue is being reviewed.", "public": True},
        "subject": "Updated ticket subject",
        "priority": "high",
        "status": "in_progress",
        "assignee_id": 12345,
    }


# Tests for PrefixedUUID


def test_default_prefix() -> None:
    """Test PrefixedUUID with default empty prefix."""
    prefixed_uuid = PrefixedUUID()
    assert prefixed_uuid.prefix == ""
    assert isinstance(prefixed_uuid.uuid, UUID)
    assert str(prefixed_uuid) == str(prefixed_uuid.uuid)


def test_custom_prefix() -> None:
    """Test PrefixedUUID with custom prefix."""

    class CustomPrefixUUID(PrefixedUUID):
        prefix = "test"

    prefixed_uuid = CustomPrefixUUID()
    assert prefixed_uuid.prefix == "test"
    assert str(prefixed_uuid).startswith("test-")
    assert str(prefixed_uuid) == f"test-{prefixed_uuid.uuid}"
    assert isinstance(prefixed_uuid.uuid, UUID)
    assert str(prefixed_uuid)[5:] == str(prefixed_uuid.uuid)


def test_from_string() -> None:
    """Test creating PrefixedUUID from string."""
    # Test with default prefix
    uuid_str = "123e4567-e89b-12d3-a456-426614174000"
    prefixed_uuid = PrefixedUUID.from_string(uuid_str)
    assert str(prefixed_uuid) == uuid_str

    # Test with custom prefix
    class CustomPrefixUUID(PrefixedUUID):
        prefix = "test"

    prefixed_str = f"test-{uuid_str}"
    prefixed_uuid = CustomPrefixUUID.from_string(prefixed_str)
    assert str(prefixed_uuid) == prefixed_str
    assert str(prefixed_uuid)[5:] == str(prefixed_uuid.uuid)

    with pytest.raises(ValueError, match="Prefix monkey does not match expected prefix test"):
        CustomPrefixUUID.from_string("monkey-123e4567-e89b-12d3-a456-426614174000")


def test_serialization() -> None:
    """Test PrefixedUUID serialization."""
    uuid = PrefixedUUID()
    assert str(uuid) == uuid.model_dump_json().strip('"')


def test_model_validation() -> None:
    """Test JSON validation and deserialization."""

    class CustomID(PrefixedUUID):
        prefix = "test"

    class TestModel(BaseModel):
        id: CustomID = Field(default_factory=CustomID)

    uuid_str = "123e4567-e89b-12d3-a456-426614174000"

    # Test with string ID
    json_data = f'{{"id": "test-{uuid_str}"}}'
    model = TestModel.model_validate_json(json_data)
    assert isinstance(model.id, CustomID)
    assert str(model.id.uuid) == uuid_str
    assert isinstance(model.id.uuid, UUID)
    assert model.id.prefix == "test"

    # Test with full representation of ID
    json_data = json.dumps(
        {
            "id": {
                "uuid": uuid_str,
            },
        },
    )
    model = TestModel.model_validate_json(json_data)
    assert isinstance(model.id, CustomID)
    assert str(model.id.uuid) == uuid_str
    assert isinstance(model.id.uuid, UUID)
    assert model.id.prefix == "test"

    json_data = f'{{"id": "monkey-{uuid_str}"}}'
    with pytest.raises(ValueError, match="Prefix monkey does not match expected prefix test"):
        TestModel.model_validate_json(json_data)

    class TestModelNoPrefix(BaseModel):
        id: PrefixedUUID

    json_data = f'{{"id": "{uuid_str}"}}'
    model = TestModelNoPrefix.model_validate_json(json_data)
    assert isinstance(model.id, PrefixedUUID)
    assert str(model.id.uuid) == uuid_str
    assert isinstance(model.id.uuid, UUID)
    assert model.id.prefix == ""


def test_hash() -> None:
    """Test PrefixedUUID hash."""
    uuid = PrefixedUUID()
    assert hash(uuid) == hash(uuid.uuid)


def test_validate_extras_dependencies() -> None:
    """Test function raises correct error when non-existing top level package is installed."""
    with mock.patch.dict(EXTRAS_GROUPS_DEPENDENCIES, {"fake-extras-package": ["fake_package.bar"]}):
        with pytest.raises(ImportError) as e:
            validate_extras_dependencies("fake-extras-package")
        assert "portia-sdk-python[fake-extras-package]" in str(e.value)


def test_validate_extras_dependencies_raise_error_false() -> None:
    """Test function doesn't raise an error when raise_error is False."""
    with mock.patch.dict(EXTRAS_GROUPS_DEPENDENCIES, {"test": ["foobarbaz"]}):
        extras_installed = validate_extras_dependencies("test", raise_error=False)
        assert extras_installed is False


def test_validate_extras_dependencies_success() -> None:
    """Test function succeeds when package is installed."""
    with mock.patch.dict(EXTRAS_GROUPS_DEPENDENCIES, {"test": ["pytest"]}):
        validate_extras_dependencies("test")


def test_singleton() -> None:
    """Test singleton decorator functionality."""

    @singleton
    class TestClass:
        def __init__(self, value: int = 0) -> None:
            self.value = value

    # Test that same instance is returned
    instance1 = TestClass(1)
    instance2 = TestClass(2)
    assert instance1 is instance2
    assert instance1.value == 1  # Value should not change on second instantiation

    # Test reset functionality
    TestClass.reset()  # type: ignore reportFunctionMemberAccess
    instance3 = TestClass(3)
    assert instance3 is not instance1
    assert instance3.value == 3
