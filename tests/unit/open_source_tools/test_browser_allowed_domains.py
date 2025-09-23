"""Unit tests for BrowserTool.allowed_domains validation."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from portia.open_source_tools.browser_tool import BrowserTool


class TestBrowserToolAllowedDomainsValidation:
    """Focused tests for validate_allowed_domains."""

    def test_validate_allowed_domains_none(self) -> None:
        """None input returns None."""
        assert BrowserTool.validate_allowed_domains(None) is None

    def test_validate_allowed_domains_valid_list(self) -> None:
        """Valid domains are normalized and preserved in order."""
        raw = ["Example.COM", " sub.example.com "]
        expected = ["example.com", "sub.example.com"]
        assert BrowserTool.validate_allowed_domains(raw) == expected

    def test_validate_allowed_domains_invalid_container_type(self) -> None:
        """Non-list container raises ValueError for validation semantics."""
        with pytest.raises(ValueError, match="allowed_domains must be a list of domain strings"):
            BrowserTool.validate_allowed_domains("not a list")  # type: ignore[arg-type]

    def test_validate_allowed_domains_invalid_item_type(self) -> None:
        """Non-string item raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain value"):
            BrowserTool.validate_allowed_domains([123])  # type: ignore[list-item]

    def test_validate_allowed_domains_empty_string_invalid(self) -> None:
        """Empty string is invalid."""
        with pytest.raises(ValueError, match="Invalid domain value"):
            BrowserTool.validate_allowed_domains([""])

    @patch("portia.open_source_tools.browser_tool.logger")
    def test_validate_allowed_domains_wildcard_warnings(self, mock_logger: MagicMock) -> None:
        """Wildcards emit warnings via logger."""
        mock_instance = Mock()
        mock_logger.return_value = mock_instance

        # Universal wildcard
        BrowserTool.validate_allowed_domains(["*"])
        msg = mock_instance.warning.call_args[0][0]
        assert "Universal wildcard '*'" in msg

        # Pattern wildcard
        mock_instance.reset_mock()
        BrowserTool.validate_allowed_domains(["*.example.com"])
        msg2 = mock_instance.warning.call_args[0][0]
        assert "Wildcard pattern '*.example.com'" in msg2
