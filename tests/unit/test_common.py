"""Tests for common classes."""

import json
from uuid import UUID

from pydantic import BaseModel, Field

from portia.common import PortiaEnum, PrefixedUUID, combine_args_kwargs


def test_portia_enum() -> None:
    """Test PortiaEnums can enumerate."""

    class MyEnum(PortiaEnum):
        OK = "OK"

    assert MyEnum.enumerate() == (("OK", "OK"),)


def test_combine_args_kwargs() -> None:
    """Test combining args and kwargs into a single dictionary."""
    result = combine_args_kwargs(1, 2, three=3, four=4)
    assert result == {"0": 1, "1": 2, "three": 3, "four": 4}


class TestPrefixedUUID:
    """Tests for PrefixedUUID."""

    def test_default_prefix(self) -> None:
        """Test PrefixedUUID with default empty prefix."""
        prefixed_uuid = PrefixedUUID()
        assert prefixed_uuid.prefix == ""
        assert isinstance(prefixed_uuid.uuid, UUID)
        assert str(prefixed_uuid) == str(prefixed_uuid.uuid)

    def test_custom_prefix(self) -> None:
        """Test PrefixedUUID with custom prefix."""
        class CustomPrefixUUID(PrefixedUUID):
            prefix = "test"

        prefixed_uuid = CustomPrefixUUID()
        assert prefixed_uuid.prefix == "test"
        assert str(prefixed_uuid).startswith("test-")
        assert str(prefixed_uuid) == f"test-{prefixed_uuid.uuid}"
        assert isinstance(prefixed_uuid.uuid, UUID)
        assert str(prefixed_uuid)[5:] == str(prefixed_uuid.uuid)

    def test_from_string(self) -> None:
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

    def test_serialization(self) -> None:
        """Test PrefixedUUID serialization."""
        uuid = PrefixedUUID()
        assert str(uuid) == uuid.model_dump_json().strip('"')

    def test_model_validation(self) -> None:
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
        json_data = json.dumps({
            "id": {
                "uuid": uuid_str,
            },
        })
        model = TestModel.model_validate_json(json_data)
        assert isinstance(model.id, CustomID)
        assert str(model.id.uuid) == uuid_str
        assert isinstance(model.id.uuid, UUID)
        assert model.id.prefix == "test"

