import pytest
from pydantic import ValidationError

from portia.open_source_tools.browser_tool import (
    BrowserTool,
    BrowserToolForUrl,
)


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


@pytest.mark.parametrize("allowed_domains", [
    ["example.com"],
    ["https://google.com", "http*://www.google.com"],
    ["https://myaccount.google.com", "https://mail.google.com"],
    ["*.example.com"],
    ["example.com", "*.test.org", "https://secure.bank.com"]
])
def test_browser_tool_allowed_domains_patterns(allowed_domains):
    tool = BrowserTool(allowed_domains=allowed_domains)
    assert tool.allowed_domains == allowed_domains
    
    tool_for_url = BrowserToolForUrl(
        url="https://example.com",
        allowed_domains=allowed_domains
    )
    assert tool_for_url.allowed_domains == allowed_domains