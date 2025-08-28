"""Test browser tool allowed_domains functionality following browser-use specification."""

import pytest
from pydantic import ValidationError
from unittest.mock import Mock

from portia.open_source_tools.browser_tool import (
    BrowserTool,
    BrowserToolForUrl,
)


def test_browser_tool_allowed_domains_exact_matching():
    tool = BrowserTool(allowed_domains=["example.com"])
    assert tool.allowed_domains == ["example.com"]




def test_browser_tool_allowed_domains_validation():
    with pytest.raises(ValidationError):
        BrowserTool(allowed_domains=[123])
    
    tool = BrowserTool(allowed_domains=["example.com", "*.test.org"])
    assert tool.allowed_domains == ["example.com", "*.test.org"]


def test_browser_tool_allowed_domains_none_default():
    tool = BrowserTool()
    assert tool.allowed_domains is None
    
    tool_explicit = BrowserTool(allowed_domains=None)
    assert tool_explicit.allowed_domains is None




def test_browser_tool_for_url_with_allowed_domains():
    url = "https://example.com"
    allowed_domains = ["example.com", "api.example.com"]
    
    tool = BrowserToolForUrl(
        url=url,
        allowed_domains=allowed_domains
    )
    
    assert tool.url == url
    assert tool.allowed_domains == allowed_domains


@pytest.mark.parametrize("allowed_domains", [
    ["example.com"],
    ["https://google.com", "http*://www.google.com"],
    ["https://myaccount.google.com", "https://mail.google.com"],
    ["*.example.com"],
    ["example.com", "*.test.org", "https://secure.bank.com"]
])
def test_browser_tool_allowed_domains_browser_use_examples(allowed_domains):
    tool = BrowserTool(allowed_domains=allowed_domains)
    assert tool.allowed_domains == allowed_domains
    
    assert isinstance(tool, BrowserTool)


def test_browser_tool_field_description_accuracy():
    tool = BrowserTool()
    field_info = tool.__class__.model_fields["allowed_domains"]
    
    description = field_info.description
    assert "exact domain matching" in description
    assert "glob patterns" in description
    assert "browser-use" in description
    assert "security" in description