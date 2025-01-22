"""Tests for common classes."""

from portia.common import PortiaEnum


def test_portia_enum() -> None:
    """Test PortiaEnums can enumerate."""

    class MyEnum(PortiaEnum):
        OK = "OK"

    assert MyEnum.enumerate() == (("OK", "OK"),)
