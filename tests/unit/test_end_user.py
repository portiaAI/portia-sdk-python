"""End user tests."""

from portia.end_user import EndUser


def test_portia_local_default_config_with_api_keys() -> None:
    """Test setting additional data."""
    end_user = EndUser(external_id="123")

    end_user.set_additional_data("other", "value")
    assert end_user.get_additional_data("other") == "value"
    end_user.remove_additional_data("other")
    assert end_user.get_additional_data("other") is None

    assert end_user.get_additional_data("not_set") is None
