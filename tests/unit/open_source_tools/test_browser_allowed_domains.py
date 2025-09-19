"""Unit tests for BrowserTool allowed_domains functionality.

This module contains focused unit tests for the allowed_domains field validation
and related functionality in isolation.
"""

from unittest.mock import Mock, patch

import pytest

from portia.open_source_tools.browser_tool import BrowserTool


class TestBrowserToolAllowedDomainsValidation:
    """Unit tests for the allowed_domains field validation."""

    def test_validate_allowed_domains_none_input(self) -> None:
        """Test that None input is handled correctly."""
        result = BrowserTool.validate_allowed_domains(None)
        assert result is None

    def test_validate_allowed_domains_valid_list(self) -> None:
        """Test validation of valid domain lists."""
        valid_domains = ["example.com", "subdomain.example.com", "another-site.org"]
        result = BrowserTool.validate_allowed_domains(valid_domains)
        assert result == ["example.com", "subdomain.example.com", "another-site.org"]

    def test_validate_allowed_domains_empty_list(self) -> None:
        """Test that empty list is handled correctly."""
        result = BrowserTool.validate_allowed_domains([])
        assert result == []

    def test_validate_allowed_domains_single_domain(self) -> None:
        """Test validation of single domain."""
        result = BrowserTool.validate_allowed_domains(["example.com"])
        assert result == ["example.com"]

    def test_validate_allowed_domains_whitespace_cleaning(self) -> None:
        """Test that whitespace is properly stripped from domains."""
        domains_with_whitespace = [" example.com ", "  subdomain.example.com  "]
        result = BrowserTool.validate_allowed_domains(domains_with_whitespace)
        assert result == ["example.com", "subdomain.example.com"]

    def test_validate_allowed_domains_case_normalization(self) -> None:
        """Test that domains are converted to lowercase."""
        mixed_case_domains = ["Example.COM", "SubDomain.Example.COM"]
        result = BrowserTool.validate_allowed_domains(mixed_case_domains)
        assert result == ["example.com", "subdomain.example.com"]

    def test_validate_allowed_domains_combined_cleaning(self) -> None:
        """Test combined whitespace and case cleaning."""
        messy_domains = [" Example.COM ", "  SubDomain.Example.COM  "]
        result = BrowserTool.validate_allowed_domains(messy_domains)
        assert result == ["example.com", "subdomain.example.com"]

    @patch("portia.open_source_tools.browser_tool.logger")
    def test_validate_allowed_domains_universal_wildcard_warning(self, mock_logger) -> None:
        """Test that universal wildcard generates appropriate warning."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        result = BrowserTool.validate_allowed_domains(["*"])

        mock_logger_instance.warning.assert_called_once_with(
            "Universal wildcard '*' allows access to ANY domain. "
            "This is extremely dangerous. Use specific domain patterns instead."
        )
        assert result == ["*"]

    @patch("portia.open_source_tools.browser_tool.logger")
    def test_validate_allowed_domains_pattern_wildcard_warning(self, mock_logger) -> None:
        """Test that pattern wildcards generate appropriate warnings."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        result = BrowserTool.validate_allowed_domains(["*.example.com"])

        mock_logger_instance.warning.assert_called_once_with(
            "Wildcard pattern '*.example.com' may match unintended domains. "
            "Per browser-use docs, be very cautious with wildcards."
        )
        assert result == ["*.example.com"]

    @patch("portia.open_source_tools.browser_tool.logger")
    def test_validate_allowed_domains_multiple_wildcards(self, mock_logger) -> None:
        """Test that multiple wildcard patterns each generate warnings."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        domains = ["*.example.com", "*.test.org", "*"]
        result = BrowserTool.validate_allowed_domains(domains)

        # Should have 3 warning calls
        assert mock_logger_instance.warning.call_count == 3
        assert result == ["*.example.com", "*.test.org", "*"]

    def test_validate_allowed_domains_invalid_input_type(self) -> None:
        """Test that non-list input raises ValueError."""
        with pytest.raises(ValueError, match="allowed_domains must be a list of domain strings"):
            BrowserTool.validate_allowed_domains("not a list")  # type: ignore

        with pytest.raises(ValueError, match="allowed_domains must be a list of domain strings"):
            BrowserTool.validate_allowed_domains(123)  # type: ignore

        with pytest.raises(ValueError, match="allowed_domains must be a list of domain strings"):
            BrowserTool.validate_allowed_domains({"domain": "example.com"})  # type: ignore

    def test_validate_allowed_domains_empty_string_domain(self) -> None:
        """Test that empty string domains raise ValueError."""
        with pytest.raises(ValueError, match="Invalid domain value: "):
            BrowserTool.validate_allowed_domains([""])

        with pytest.raises(ValueError, match="Invalid domain value:"):
            BrowserTool.validate_allowed_domains(["   "])  # whitespace only

    def test_validate_allowed_domains_non_string_domain(self) -> None:
        """Test that non-string domain values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid domain value: 123"):
            BrowserTool.validate_allowed_domains([123])  # type: ignore

        with pytest.raises(ValueError, match="Invalid domain value: None"):
            BrowserTool.validate_allowed_domains([None])  # type: ignore

        with pytest.raises(ValueError, match="Invalid domain value:"):
            BrowserTool.validate_allowed_domains([["nested", "list"]])  # type: ignore

    def test_validate_allowed_domains_mixed_valid_invalid(self) -> None:
        """Test that validation fails on first invalid domain in mixed list."""
        # Should fail on the empty string, even though first domain is valid
        with pytest.raises(ValueError, match="Invalid domain value: "):
            BrowserTool.validate_allowed_domains(["example.com", ""])

        # Should fail on the non-string, even though first domain is valid
        with pytest.raises(ValueError, match="Invalid domain value: 123"):
            BrowserTool.validate_allowed_domains(["example.com", 123])  # type: ignore

    def test_validate_allowed_domains_special_characters(self) -> None:
        """Test domains with special characters are handled correctly."""
        special_domains = [
            "example-site.com",
            "site_with_underscores.org",
            "123numeric.com",
            "sub.domain.example.com",
        ]
        result = BrowserTool.validate_allowed_domains(special_domains)
        expected = [domain.lower() for domain in special_domains]
        assert result == expected

    def test_validate_allowed_domains_international_domains(self) -> None:
        """Test that international domain formats work."""
        # Note: These would be punycode in real usage, but testing the validation logic
        international_domains = ["example.co.uk", "site.com.au", "test.de"]
        result = BrowserTool.validate_allowed_domains(international_domains)
        assert result == international_domains

    def test_validate_allowed_domains_edge_cases(self) -> None:
        """Test edge cases in domain validation."""
        # Very long domain name
        long_domain = "a" * 60 + ".com"
        result = BrowserTool.validate_allowed_domains([long_domain])
        assert result == [long_domain]

        # Single character domain parts
        short_domains = ["a.b", "x.co"]
        result = BrowserTool.validate_allowed_domains(short_domains)
        assert result == short_domains

    def test_validate_allowed_domains_preserves_order(self) -> None:
        """Test that domain order is preserved."""
        domains = ["z.com", "a.com", "m.com"]
        result = BrowserTool.validate_allowed_domains(domains)
        assert result == ["z.com", "a.com", "m.com"]  # Same order

    def test_validate_allowed_domains_duplicate_handling(self) -> None:
        """Test that duplicate domains are preserved (not deduplicated)."""
        domains = ["example.com", "example.com", "test.org"]
        result = BrowserTool.validate_allowed_domains(domains)
        assert result == ["example.com", "example.com", "test.org"]

    @patch("portia.open_source_tools.browser_tool.logger")
    def test_validate_allowed_domains_no_warning_for_normal_domains(self, mock_logger) -> None:
        """Test that normal domains don't generate warnings."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        normal_domains = ["example.com", "test.org", "site-name.net"]
        result = BrowserTool.validate_allowed_domains(normal_domains)

        # Should not have any warning calls
        mock_logger_instance.warning.assert_not_called()
        assert result == normal_domains


class TestBrowserToolAllowedDomainsFieldIntegration:
    """Unit tests for the allowed_domains field integration with BrowserTool."""

    def test_browser_tool_initialization_default(self) -> None:
        """Test that BrowserTool initializes with None allowed_domains by default."""
        tool = BrowserTool()
        assert tool.allowed_domains is None

    def test_browser_tool_initialization_with_allowed_domains(self) -> None:
        """Test that BrowserTool can be initialized with allowed_domains."""
        domains = ["example.com", "test.org"]
        tool = BrowserTool(allowed_domains=domains)
        assert tool.allowed_domains == domains

    def test_browser_tool_field_validation_on_init(self) -> None:
        """Test that field validation runs during initialization."""
        # Valid case
        tool = BrowserTool(allowed_domains=["example.com"])
        assert tool.allowed_domains == ["example.com"]

        # Invalid case should raise during init
        with pytest.raises(ValueError):
            BrowserTool(allowed_domains="invalid")  # type: ignore

    def test_browser_tool_allowed_domains_field_descriptor(self) -> None:
        """Test that allowed_domains field has correct descriptor properties."""
        # Test that the field exists and has the right properties
        from pydantic.fields import FieldInfo

        # Get the field info
        field_info = BrowserTool.model_fields.get("allowed_domains")
        assert field_info is not None
        assert isinstance(field_info, FieldInfo)

        # Check default value
        assert field_info.default is None

        # Check description
        expected_description = (
            "List of allowed domains for browser navigation. "
            "If specified, navigation will be restricted to these domains only."
        )
        assert field_info.description == expected_description

    def test_browser_tool_model_dump_includes_allowed_domains(self) -> None:
        """Test that model serialization includes allowed_domains."""
        domains = ["example.com", "test.org"]
        tool = BrowserTool(allowed_domains=domains)

        model_dict = tool.model_dump()
        assert "allowed_domains" in model_dict
        assert model_dict["allowed_domains"] == domains

    def test_browser_tool_model_dump_none_allowed_domains(self) -> None:
        """Test that model serialization handles None allowed_domains."""
        tool = BrowserTool(allowed_domains=None)

        model_dict = tool.model_dump()
        assert "allowed_domains" in model_dict
        assert model_dict["allowed_domains"] is None
