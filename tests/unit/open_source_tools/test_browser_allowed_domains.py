"""Test browser tool allowed_domains functionality following browser-use specification."""

import pytest
from pydantic import ValidationError
from unittest.mock import Mock

from portia.open_source_tools.browser_tool import (
    BrowserTool,
    BrowserToolForUrl,
)


class TestBrowserToolAllowedDomains:
    """Test allowed_domains functionality per browser-use documentation."""

    def test_browser_tool_allowed_domains_exact_matching(self):
        """Test exact domain matching as per browser-use docs."""
        # Example from docs: allowed_domains=['example.com'] only matches https://example.com/*
        tool = BrowserTool(allowed_domains=["example.com"])
        assert tool.allowed_domains == ["example.com"]

    def test_browser_tool_allowed_domains_with_schemes(self):
        """Test domain specification with schemes."""
        # Example from docs: multiple schemes and subdomains
        allowed_domains = [
            "https://google.com",
            "http*://www.google.com", 
            "https://myaccount.google.com"
        ]
        tool = BrowserTool(allowed_domains=allowed_domains)
        assert tool.allowed_domains == allowed_domains

    def test_browser_tool_allowed_domains_validation(self):
        """Test basic validation of allowed_domains format."""
        # Test invalid domain types should raise validation errors (Pydantic handles this)
        with pytest.raises(ValidationError):
            BrowserTool(allowed_domains=[123])
        
        # Test valid domains work
        tool = BrowserTool(allowed_domains=["example.com", "*.test.org"])
        assert tool.allowed_domains == ["example.com", "*.test.org"]

    def test_browser_tool_allowed_domains_none_default(self):
        """Test that None allowed_domains allows all domains (default behavior)."""
        tool = BrowserTool()
        assert tool.allowed_domains is None
        
        tool_explicit = BrowserTool(allowed_domains=None)
        assert tool_explicit.allowed_domains is None

    def test_browser_tool_allowed_domains_stored_correctly(self):
        """Test that allowed_domains is correctly stored on the tool."""
        allowed_domains = ["example.com", "*.test.org"]
        tool = BrowserTool(allowed_domains=allowed_domains)
        
        # Test allowed_domains are stored correctly
        assert tool.allowed_domains == allowed_domains
        
        # Test tool can be created successfully
        assert isinstance(tool, BrowserTool)

    def test_browser_tool_for_url_with_allowed_domains(self):
        """Test BrowserToolForUrl properly handles allowed_domains parameter."""
        url = "https://example.com"
        allowed_domains = ["example.com", "api.example.com"]
        
        tool = BrowserToolForUrl(
            url=url,
            allowed_domains=allowed_domains
        )
        
        assert tool.url == url
        assert tool.allowed_domains == allowed_domains

    def test_browser_tool_allowed_domains_browser_use_examples(self):
        """Test various allowed_domains formats from browser-use documentation."""
        # Test cases from browser-use docs
        test_cases = [
            # Exact domain matching
            ["example.com"],
            
            # Multiple domains with schemes  
            ["https://google.com", "http*://www.google.com"],
            
            # Subdomain specification
            ["https://myaccount.google.com", "https://mail.google.com"],
            
            # Wildcard patterns (use with caution)
            ["*.example.com"],
            
            # Mixed patterns
            ["example.com", "*.test.org", "https://secure.bank.com"]
        ]
        
        for allowed_domains in test_cases:
            tool = BrowserTool(allowed_domains=allowed_domains)
            assert tool.allowed_domains == allowed_domains
            
            # Ensure tool can be created without errors
            assert isinstance(tool, BrowserTool)

    def test_browser_tool_field_description_accuracy(self):
        """Test that the allowed_domains field description matches browser-use behavior."""
        tool = BrowserTool()
        field_info = tool.__class__.model_fields["allowed_domains"]
        
        # Verify description mentions key browser-use concepts
        description = field_info.description
        assert "exact domain matching" in description
        assert "glob patterns" in description
        assert "browser-use" in description
        assert "security" in description