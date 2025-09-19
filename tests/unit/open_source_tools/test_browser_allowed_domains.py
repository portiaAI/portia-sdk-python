"""Unit tests for BrowserTool allowed_domains functionality.""""""Unit tests for BrowserTool allowed_domains functionality."""Unit tests for BrowserTool allowed_domains functiona    @patch("por    @patch("por    @patch("portia.open_source_tools.browser_tool.logger")



import pytest    def test_validate_allowed_domains    @patch("portia.open_source_tools.browser_tool.logger")

from unittest.mock import Mock, MagicMock, patch

This module contains focused unit tests for the allowed_domains field validation    def test_validate_allowed_domains_no_warning_for_normal_domains(self, mock_logger: MagicMock) -> None:ultiple_wildcards(self, mock_logger: MagicMock) -> None:a.open_source_tools.browser_tool.logger")

from portia.open_source_tools.browser_tool import BrowserTool

and related functionality in isolation.    def test_validate_allowed_domains_pattern_wildcard_warning(self, mock_logger: MagicMock) -> None:a.open_source_tools.browser_tool.logger")



class TestBrowserToolAllowedDomainsValidation:"""    def test_validate_allowed_domains_universal_wildcard_warning(self, mock_logger: MagicMock) -> None:ty.

    """Test the allowed_domains validation method in isolation."""



    def test_validate_allowed_domains_valid_domains(self) -> None:

        """Test that valid domain lists pass validation."""import pytestThis module contains focused unit tests for the allowed_domains field validation

        valid_domains = ["example.com", "google.com", "github.com"]

        result = BrowserTool.validate_allowed_domains(valid_domains)from unittest.mock import Mock, MagicMock, patchand related functionality in isolation.

        assert result == valid_domains

from typing import Any"""

    def test_validate_allowed_domains_empty_list(self) -> None:

        """Test that empty list is valid."""

        result = BrowserTool.validate_allowed_domains([])

        assert result == []from portia.open_source_tools.browser_tool import BrowserToolimport pytest



    def test_validate_allowed_domains_case_normalization(self) -> None:from unittest.mock import Mock, MagicMock, patch

        """Test that domains are normalized to lowercase."""

        domains = ["Example.COM", "Google.com", "GITHUB.COM"]from typing import Any

        result = BrowserTool.validate_allowed_domains(domains)

        assert result == ["example.com", "google.com", "github.com"]class TestBrowserToolAllowedDomainsValidation:



    @patch("portia.open_source_tools.browser_tool.logger")    """Test the allowed_domains validation method in isolation."""from portia.open_source_tools.browser_tool import BrowserTool

    def test_validate_allowed_domains_universal_wildcard_warning(

        self, mock_logger: MagicMock

    ) -> None:

        """Test that universal wildcard generates appropriate warning."""    def test_validate_allowed_domains_valid_domains(self) -> None:

        mock_logger_instance = Mock()

        mock_logger.return_value = mock_logger_instance        """Test that valid domain lists pass validation."""class TestBrowserToolAllowedDomainsValidation:



        domains = ["*"]        valid_domains = ["example.com", "google.com", "github.com"]    """Unit tests for the allowed_domains field validation."""

        result = BrowserTool.validate_allowed_domains(domains)

        assert result == ["*"]        result = BrowserTool.validate_allowed_domains(valid_domains)



        # Verify warning was logged        assert result == valid_domains    def test_validate_allowed_domains_none_input(self):

        mock_logger_instance.warning.assert_called_once()

        """Test that None input is handled correctly."""

    @patch("portia.open_source_tools.browser_tool.logger")

    def test_validate_allowed_domains_pattern_wildcard_warning(    def test_validate_allowed_domains_empty_list(self) -> None:        result = BrowserTool.validate_allowed_domains(None)

        self, mock_logger: MagicMock

    ) -> None:        """Test that empty list is valid."""        assert result is None

        """Test that pattern wildcards generate appropriate warnings."""

        mock_logger_instance = Mock()        result = BrowserTool.validate_allowed_domains([])

        mock_logger.return_value = mock_logger_instance

        assert result == []    def test_validate_allowed_domains_valid_list(self):

        domains = ["*.example.com"]

        result = BrowserTool.validate_allowed_domains(domains)        """Test validation of valid domain lists."""

        assert result == ["*.example.com"]

    def test_validate_allowed_domains_single_domain(self) -> None:        valid_domains = ["example.com", "subdomain.example.com", "another-site.org"]

        # Verify warning was logged

        mock_logger_instance.warning.assert_called_once()        """Test that single domain list works."""        result = BrowserTool.validate_allowed_domains(valid_domains)



    @patch("portia.open_source_tools.browser_tool.logger")        result = BrowserTool.validate_allowed_domains(["example.com"])        assert result == ["example.com", "subdomain.example.com", "another-site.org"]

    def test_validate_allowed_domains_multiple_wildcards(

        self, mock_logger: MagicMock        assert result == ["example.com"]

    ) -> None:

        """Test that multiple wildcard patterns each generate warnings."""    def test_validate_allowed_domains_empty_list(self):

        mock_logger_instance = Mock()

        mock_logger.return_value = mock_logger_instance    def test_validate_allowed_domains_case_normalization(self) -> None:        """Test that empty list is handled correctly."""



        domains = ["*.example.com", "*", "*.github.com"]        """Test that domains are normalized to lowercase."""        result = BrowserTool.validate_allowed_domains([])

        result = BrowserTool.validate_allowed_domains(domains)

        assert result == ["*.example.com", "*", "*.github.com"]        domains = ["Example.COM", "Google.com", "GITHUB.COM"]        assert result == []



        # Should have 3 warning calls - one for each wildcard        result = BrowserTool.validate_allowed_domains(domains)

        assert mock_logger_instance.warning.call_count == 3

        assert result == ["example.com", "google.com", "github.com"]    def test_validate_allowed_domains_single_domain(self):

    def test_validate_allowed_domains_non_list_input(self) -> None:

        """Test that non-list input raises ValueError."""        """Test validation of single domain."""

        with pytest.raises(

            ValueError, match="allowed_domains must be a list of domain strings"    def test_validate_allowed_domains_whitespace_cleanup(self) -> None:        result = BrowserTool.validate_allowed_domains(["example.com"])

        ):

            BrowserTool.validate_allowed_domains("not a list")  # type: ignore[arg-type]        """Test that whitespace is stripped from domains."""        assert result == ["example.com"]



        with pytest.raises(        domains = [" example.com ", "  google.com  ", "\tgithub.com\n"]

            ValueError, match="allowed_domains must be a list of domain strings"

        ):        result = BrowserTool.validate_allowed_domains(domains)    def test_validate_allowed_domains_whitespace_cleaning(self):

            BrowserTool.validate_allowed_domains(123)  # type: ignore[arg-type]

        assert result == ["example.com", "google.com", "github.com"]        """Test that whitespace is properly stripped from domains."""

        with pytest.raises(

            ValueError, match="allowed_domains must be a list of domain strings"        domains_with_whitespace = [" example.com ", "  subdomain.example.com  "]

        ):

            BrowserTool.validate_allowed_domains(    def test_validate_allowed_domains_subdomain_support(self) -> None:        result = BrowserTool.validate_allowed_domains(domains_with_whitespace)

                {"domain": "example.com"}

            )  # type: ignore[arg-type]        """Test that subdomains are supported."""        assert result == ["example.com", "subdomain.example.com"]



    def test_validate_allowed_domains_non_string_domain_values(self) -> None:        domains = ["api.example.com", "subdomain.github.com", "www.google.com"]

        """Test that non-string domain values raise ValueError."""

        with pytest.raises(ValueError, match="Invalid domain value: 123"):        result = BrowserTool.validate_allowed_domains(domains)    def test_validate_allowed_domains_case_normalization(self):

            BrowserTool.validate_allowed_domains([123])  # type: ignore[list-item]

        assert result == domains        """Test that domains are converted to lowercase."""

        with pytest.raises(ValueError, match="Invalid domain value: None"):

            BrowserTool.validate_allowed_domains([None])  # type: ignore[list-item]        mixed_case_domains = ["Example.COM", "SubDomain.Example.COM"]



    @patch("portia.open_source_tools.browser_tool.logger")    def test_validate_allowed_domains_wildcard_patterns(self) -> None:        result = BrowserTool.validate_allowed_domains(mixed_case_domains)

    def test_validate_allowed_domains_no_warning_for_normal_domains(

        self, mock_logger: MagicMock        """Test that wildcard patterns are valid."""        assert result == ["example.com", "subdomain.example.com"]

    ) -> None:

        """Test that normal domains don't generate warnings."""        domains = ["*.example.com", "*.github.com", "sub.*.google.com"]

        mock_logger_instance = Mock()

        mock_logger.return_value = mock_logger_instance        result = BrowserTool.validate_allowed_domains(domains)    def test_validate_allowed_domains_combined_cleaning(self):



        domains = ["example.com", "google.com", "github.com"]        assert result == domains        """Test combined whitespace and case cleaning."""

        result = BrowserTool.validate_allowed_domains(domains)

        assert result == domains        messy_domains = [" Example.COM ", "  SubDomain.Example.COM  "]



        # Should not have any warning calls for normal domains    def test_validate_allowed_domains_combined_cleanup(self) -> None:        result = BrowserTool.validate_allowed_domains(messy_domains)

        mock_logger_instance.warning.assert_not_called()

        """Test combined whitespace and case cleaning."""        assert result == ["example.com", "subdomain.example.com"]



class TestBrowserToolAllowedDomainsFieldIntegration:        messy_domains = [" Example.COM ", "  SubDomain.Example.COM  "]

    """Test the allowed_domains field integration with the BrowserTool class."""

        result = BrowserTool.validate_allowed_domains(messy_domains)    @patch('portia.open_source_tools.browser_tool.logger')

    def test_browser_tool_allowed_domains_init_valid(self) -> None:

        """Test that BrowserTool initializes correctly with valid allowed_domains."""        assert result == ["example.com", "subdomain.example.com"]    def test_validate_allowed_domains_universal_wildcard_warning(self, mock_logger: Any) -> None:

        tool = BrowserTool(allowed_domains=["example.com", "google.com"])

        assert tool.allowed_domains == ["example.com", "google.com"]        """Test that universal wildcard generates appropriate warning."""



    def test_browser_tool_allowed_domains_init_none(self) -> None:    @patch("portia.open_source_tools.browser_tool.logger")        mock_logger_instance = Mock()

        """Test that BrowserTool initializes correctly with None allowed_domains."""

        tool = BrowserTool()    def test_validate_allowed_domains_universal_wildcard_warning(self, mock_logger: MagicMock) -> None:        mock_logger.return_value = mock_logger_instance

        assert tool.allowed_domains is None

        """Test that universal wildcard generates appropriate warning."""        

    def test_browser_tool_allowed_domains_init_invalid(self) -> None:

        """Test that BrowserTool raises ValueError for invalid allowed_domains."""        mock_logger_instance = Mock()        result = BrowserTool.validate_allowed_domains(["*"])

        # Valid case

        tool = BrowserTool(allowed_domains=["example.com"])        mock_logger.return_value = mock_logger_instance        

        assert tool.allowed_domains == ["example.com"]

        mock_logger_instance.warning.assert_called_once_with(

        # Invalid case should raise during init

        with pytest.raises(ValueError, match="allowed_domains must be a list"):        domains = ["*"]            "Universal wildcard '*' allows access to ANY domain. "

            BrowserTool(allowed_domains="invalid")  # type: ignore[arg-type]

        result = BrowserTool.validate_allowed_domains(domains)            "This is extremely dangerous. Use specific domain patterns instead."

    def test_browser_tool_allowed_domains_field_descriptor(self) -> None:

        """Test that allowed_domains field has proper metadata."""        assert result == ["*"]        )

        from pydantic.fields import FieldInfo

        assert result == ["*"]

        # Get field info through model fields

        field_info = BrowserTool.model_fields.get("allowed_domains")        # Verify warning was logged

        assert field_info is not None

        assert isinstance(field_info, FieldInfo)        mock_logger_instance.warning.assert_called_once()    @patch('portia.open_source_tools.browser_tool.logger')



        # Check field properties        warning_call = mock_logger_instance.warning.call_args[0][0]    def test_validate_allowed_domains_pattern_wildcard_warning(self, mock_logger: Any) -> None:

        assert field_info.description is not None

        assert "domains" in field_info.description.lower()        assert "Universal wildcard '*' allows all domains" in warning_call        """Test that pattern wildcards generate appropriate warnings."""

        mock_logger_instance = Mock()

    @patch("portia.open_source_tools.browser_tool.logger")        mock_logger.return_value = mock_logger_instance

    def test_validate_allowed_domains_pattern_wildcard_warning(self, mock_logger: MagicMock) -> None:        

        """Test that pattern wildcards generate appropriate warnings."""        result = BrowserTool.validate_allowed_domains(["*.example.com"])

        mock_logger_instance = Mock()        

        mock_logger.return_value = mock_logger_instance        mock_logger_instance.warning.assert_called_once_with(

            "Wildcard pattern '*.example.com' may match unintended domains. "

        domains = ["*.example.com"]            "Per browser-use docs, be very cautious with wildcards."

        result = BrowserTool.validate_allowed_domains(domains)        )

        assert result == ["*.example.com"]        assert result == ["*.example.com"]



        # Verify warning was logged    @patch('portia.open_source_tools.browser_tool.logger')

        mock_logger_instance.warning.assert_called_once()    def test_validate_allowed_domains_multiple_wildcards(self, mock_logger: Any) -> None:

        warning_call = mock_logger_instance.warning.call_args[0][0]        """Test that multiple wildcard patterns each generate warnings."""

        assert "Wildcard pattern" in warning_call        mock_logger_instance = Mock()

        assert "*.example.com" in warning_call        mock_logger.return_value = mock_logger_instance

        

    @patch("portia.open_source_tools.browser_tool.logger")        domains = ["*.example.com", "*.test.org", "*"]

    def test_validate_allowed_domains_multiple_wildcards(self, mock_logger: MagicMock) -> None:        result = BrowserTool.validate_allowed_domains(domains)

        """Test that multiple wildcard patterns each generate warnings."""        

        mock_logger_instance = Mock()        # Should have 3 warning calls

        mock_logger.return_value = mock_logger_instance        assert mock_logger_instance.warning.call_count == 3

        assert result == ["*.example.com", "*.test.org", "*"]

        domains = ["*.example.com", "*", "*.github.com"]

        result = BrowserTool.validate_allowed_domains(domains)    def test_validate_allowed_domains_invalid_input_type(self):

        assert result == ["*.example.com", "*", "*.github.com"]        """Test that non-list input raises ValueError."""

        with pytest.raises(ValueError, match="allowed_domains must be a list of domain strings"):

        # Should have 3 warning calls - one for each wildcard            BrowserTool.validate_allowed_domains("not a list")  # type: ignore[arg-type][arg-type]

        assert mock_logger_instance.warning.call_count == 3

        with pytest.raises(ValueError, match="allowed_domains must be a list of domain strings"):

    def test_validate_allowed_domains_non_list_input(self) -> None:            BrowserTool.validate_allowed_domains(123)  # type: ignore[arg-type][arg-type]

        """Test that non-list input raises ValueError."""

        with pytest.raises(ValueError, match="allowed_domains must be a list of domain strings"):        with pytest.raises(ValueError, match="allowed_domains must be a list of domain strings"):

            BrowserTool.validate_allowed_domains("not a list")  # type: ignore[arg-type]            BrowserTool.validate_allowed_domains({"domain": "example.com"})  # type: ignore[arg-type][arg-type]



        with pytest.raises(ValueError, match="allowed_domains must be a list of domain strings"):    def test_validate_allowed_domains_empty_string_domain(self):

            BrowserTool.validate_allowed_domains(123)  # type: ignore[arg-type]        """Test that empty string domains raise ValueError."""

        with pytest.raises(ValueError, match="Invalid domain value: "):

        with pytest.raises(ValueError, match="allowed_domains must be a list of domain strings"):            BrowserTool.validate_allowed_domains([""])

            BrowserTool.validate_allowed_domains({"domain": "example.com"})  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Invalid domain value:"):

    def test_validate_allowed_domains_empty_string_domain(self) -> None:            BrowserTool.validate_allowed_domains(["   "])  # whitespace only

        """Test that empty string domains raise ValueError."""

        with pytest.raises(ValueError, match="Invalid domain value: ''"):    def test_validate_allowed_domains_non_string_domain(self):

            BrowserTool.validate_allowed_domains([""])        """Test that non-string domain values raise ValueError."""

        with pytest.raises(ValueError, match="Invalid domain value: 123"):

        with pytest.raises(ValueError, match="Invalid domain value: '   '"):            BrowserTool.validate_allowed_domains([123])  # type: ignore[list-item][list-item]

            BrowserTool.validate_allowed_domains(["   "])

        with pytest.raises(ValueError, match="Invalid domain value: None"):

        with pytest.raises(ValueError, match="Invalid domain value: '\t\n'"):            BrowserTool.validate_allowed_domains([None])  # type: ignore[list-item][list-item]

            BrowserTool.validate_allowed_domains(["\t\n"])

        with pytest.raises(ValueError, match="Invalid domain value:"):

    def test_validate_allowed_domains_non_string_domain_values(self) -> None:            BrowserTool.validate_allowed_domains([["nested", "list"]])  # type: ignore[list-item][list-item]

        """Test that non-string domain values raise ValueError."""

        with pytest.raises(ValueError, match="Invalid domain value: 123"):    def test_validate_allowed_domains_mixed_valid_invalid(self):

            BrowserTool.validate_allowed_domains([123])  # type: ignore[list-item]        """Test that validation fails on first invalid domain in mixed list."""

        # Should fail on the empty string, even though first domain is valid

        with pytest.raises(ValueError, match="Invalid domain value: None"):        with pytest.raises(ValueError, match="Invalid domain value: "):

            BrowserTool.validate_allowed_domains([None])  # type: ignore[list-item]            BrowserTool.validate_allowed_domains(["example.com", ""])



        with pytest.raises(ValueError, match="Invalid domain value:"):        # Should fail on the non-string, even though first domain is valid  

            BrowserTool.validate_allowed_domains([["nested", "list"]])  # type: ignore[list-item]        with pytest.raises(ValueError, match="Invalid domain value: 123"):

            BrowserTool.validate_allowed_domains(["example.com", 123])  # type: ignore[list-item][list-item]

    def test_validate_allowed_domains_mixed_valid_invalid(self) -> None:

        """Test that mixed valid/invalid domains fail on first invalid."""    def test_validate_allowed_domains_special_characters(self):

        # Should fail on the non-string, even though first domain is valid        """Test domains with special characters are handled correctly."""

        with pytest.raises(ValueError, match="Invalid domain value: 123"):        special_domains = [

            BrowserTool.validate_allowed_domains(["example.com", 123])  # type: ignore[list-item]            "example-site.com",

            "site_with_underscores.org", 

    def test_validate_allowed_domains_special_characters(self) -> None:            "123numeric.com",

        """Test domains with special characters."""            "sub.domain.example.com"

        # These should be valid        ]

        valid_domains = [        result = BrowserTool.validate_allowed_domains(special_domains)

            "example-site.com",        expected = [domain.lower() for domain in special_domains]

            "test_domain.org",        assert result == expected

            "sub-domain.example.com",

            "123.example.com",    def test_validate_allowed_domains_international_domains(self):

            "example.co.uk"        """Test that international domain formats work."""

        ]        # Note: These would be punycode in real usage, but testing the validation logic

        result = BrowserTool.validate_allowed_domains(valid_domains)        international_domains = ["example.co.uk", "site.com.au", "test.de"]

        assert result == valid_domains        result = BrowserTool.validate_allowed_domains(international_domains)

        assert result == international_domains

    def test_validate_allowed_domains_localhost_support(self) -> None:

        """Test that localhost and local development domains are supported."""    def test_validate_allowed_domains_edge_cases(self):

        local_domains = [        """Test edge cases in domain validation."""

            "localhost",        # Very long domain name

            "127.0.0.1",        long_domain = "a" * 60 + ".com"

            "localhost:8080",        result = BrowserTool.validate_allowed_domains([long_domain])

            "127.0.0.1:3000",        assert result == [long_domain]

            "dev.local",

            "test.localhost"        # Single character domain parts

        ]        short_domains = ["a.b", "x.co"]

        result = BrowserTool.validate_allowed_domains(local_domains)        result = BrowserTool.validate_allowed_domains(short_domains)

        assert result == local_domains        assert result == short_domains



    def test_validate_allowed_domains_port_numbers(self) -> None:    def test_validate_allowed_domains_preserves_order(self):

        """Test domains with port numbers."""        """Test that domain order is preserved."""

        domains_with_ports = [        domains = ["z.com", "a.com", "m.com"]

            "example.com:8080",        result = BrowserTool.validate_allowed_domains(domains)

            "api.example.com:443",        assert result == ["z.com", "a.com", "m.com"]  # Same order

            "localhost:3000",

            "127.0.0.1:8000"    def test_validate_allowed_domains_duplicate_handling(self):

        ]        """Test that duplicate domains are preserved (not deduplicated)."""

        result = BrowserTool.validate_allowed_domains(domains_with_ports)        domains = ["example.com", "example.com", "test.org"]

        assert result == domains_with_ports        result = BrowserTool.validate_allowed_domains(domains)

        assert result == ["example.com", "example.com", "test.org"]

    def test_validate_allowed_domains_duplicate_removal(self) -> None:

        """Test that duplicate domains are removed while preserving order."""    @patch('portia.open_source_tools.browser_tool.logger')

        domains_with_duplicates = [    def test_validate_allowed_domains_no_warning_for_normal_domains(self, mock_logger: Any) -> None:

            "example.com",        """Test that normal domains don't generate warnings."""

            "google.com",        mock_logger_instance = Mock()

            "example.com",  # duplicate        mock_logger.return_value = mock_logger_instance

            "github.com",        

            "google.com",   # duplicate        normal_domains = ["example.com", "test.org", "site-name.net"]

            "example.com"   # duplicate        result = BrowserTool.validate_allowed_domains(normal_domains)

        ]        

        result = BrowserTool.validate_allowed_domains(domains_with_duplicates)        # Should not have any warning calls

        assert result == ["example.com", "google.com", "github.com"]        mock_logger_instance.warning.assert_not_called()

        assert result == normal_domains

    @patch("portia.open_source_tools.browser_tool.logger")

    def test_validate_allowed_domains_no_warning_for_normal_domains(self, mock_logger: MagicMock) -> None:

        """Test that normal domains don't generate warnings."""class TestBrowserToolAllowedDomainsFieldIntegration:

        mock_logger_instance = Mock()    """Unit tests for the allowed_domains field integration with BrowserTool."""

        mock_logger.return_value = mock_logger_instance

    def test_browser_tool_initialization_default(self):

        domains = ["example.com", "google.com", "github.com"]        """Test that BrowserTool initializes with None allowed_domains by default."""

        result = BrowserTool.validate_allowed_domains(domains)        tool = BrowserTool()

        assert result == domains        assert tool.allowed_domains is None



        # Should not have any warning calls for normal domains    def test_browser_tool_initialization_with_allowed_domains(self):

        mock_logger_instance.warning.assert_not_called()        """Test that BrowserTool can be initialized with allowed_domains."""

        domains = ["example.com", "test.org"]

        tool = BrowserTool(allowed_domains=domains)

class TestBrowserToolAllowedDomainsFieldIntegration:        assert tool.allowed_domains == domains

    """Test the allowed_domains field integration with the BrowserTool class."""

    def test_browser_tool_field_validation_on_init(self):

    def test_browser_tool_allowed_domains_init_valid(self) -> None:        """Test that field validation runs during initialization."""

        """Test that BrowserTool initializes correctly with valid allowed_domains."""        # Valid case

        tool = BrowserTool(allowed_domains=["example.com", "google.com"])        tool = BrowserTool(allowed_domains=["example.com"])

        assert tool.allowed_domains == ["example.com", "google.com"]        assert tool.allowed_domains == ["example.com"]



    def test_browser_tool_allowed_domains_init_empty(self) -> None:        # Invalid case should raise during init

        """Test that BrowserTool initializes correctly with empty allowed_domains."""        with pytest.raises(ValueError, match="allowed_domains must be a list"):

        tool = BrowserTool(allowed_domains=[])            BrowserTool(allowed_domains="invalid")  # type: ignore[arg-type]

        assert tool.allowed_domains == []

    def test_browser_tool_allowed_domains_field_descriptor(self):

    def test_browser_tool_allowed_domains_init_none(self) -> None:        """Test that allowed_domains field has correct descriptor properties."""

        """Test that BrowserTool initializes correctly with None allowed_domains."""        # Test that the field exists and has the right properties

        tool = BrowserTool()        from pydantic.fields import FieldInfo

        assert tool.allowed_domains is None        

        # Get the field info

    def test_browser_tool_allowed_domains_init_invalid(self) -> None:        field_info = BrowserTool.model_fields.get('allowed_domains')

        """Test that BrowserTool raises ValueError for invalid allowed_domains."""        assert field_info is not None

        # Valid case        assert isinstance(field_info, FieldInfo)

        tool = BrowserTool(allowed_domains=["example.com"])        

        assert tool.allowed_domains == ["example.com"]        # Check default value

        assert field_info.default is None

        # Invalid case should raise during init        

        with pytest.raises(ValueError, match="allowed_domains must be a list"):        # Check description

            BrowserTool(allowed_domains="invalid")  # type: ignore[arg-type]        expected_description = (

            "List of allowed domains for browser navigation. "

    def test_browser_tool_allowed_domains_field_descriptor(self) -> None:            "If specified, navigation will be restricted to these domains only."

        """Test that allowed_domains field has proper metadata."""        )

        from pydantic.fields import FieldInfo        assert field_info.description == expected_description



        # Get field info through model fields    def test_browser_tool_model_dump_includes_allowed_domains(self):

        field_info = BrowserTool.model_fields.get("allowed_domains")        """Test that model serialization includes allowed_domains."""

        assert field_info is not None        domains = ["example.com", "test.org"] 

        assert isinstance(field_info, FieldInfo)        tool = BrowserTool(allowed_domains=domains)

        

        # Check field properties        model_dict = tool.model_dump()

        assert field_info.description is not None        assert 'allowed_domains' in model_dict

        assert "domains" in field_info.description.lower()        assert model_dict['allowed_domains'] == domains



    def test_browser_tool_allowed_domains_pydantic_validation(self) -> None:    def test_browser_tool_model_dump_none_allowed_domains(self):

        """Test that Pydantic validation works correctly."""        """Test that model serialization handles None allowed_domains."""

        # Valid initialization        tool = BrowserTool(allowed_domains=None)

        tool = BrowserTool(allowed_domains=["example.com"])        

        assert tool.allowed_domains == ["example.com"]        model_dict = tool.model_dump()

        assert 'allowed_domains' in model_dict

        # Invalid types should be caught by Pydantic        assert model_dict['allowed_domains'] is None
        with pytest.raises(ValueError):
            BrowserTool(allowed_domains="not a list")  # type: ignore[arg-type]

    def test_browser_tool_allowed_domains_serialization(self) -> None:
        """Test that allowed_domains serializes correctly."""
        tool = BrowserTool(allowed_domains=["example.com", "google.com"])

        # Test dict conversion
        tool_dict = tool.model_dump()
        assert "allowed_domains" in tool_dict
        assert tool_dict["allowed_domains"] == ["example.com", "google.com"]

        # Test JSON serialization
        tool_json = tool.model_dump_json()
        assert '"allowed_domains":["example.com","google.com"]' in tool_json.replace(" ", "")

    def test_browser_tool_allowed_domains_deserialization(self) -> None:
        """Test that allowed_domains deserializes correctly."""
        tool_data = {
            "allowed_domains": ["example.com", "google.com"]
        }

        tool = BrowserTool(**tool_data)
        assert tool.allowed_domains == ["example.com", "google.com"]

        # Test with model_validate
        tool2 = BrowserTool.model_validate(tool_data)
        assert tool2.allowed_domains == ["example.com", "google.com"]