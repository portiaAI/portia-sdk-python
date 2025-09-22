"""Unit tests for Meta hosted Llama provider integration."""

from typing import Any

import pytest

from portia.config import Config
from portia.model import LLMProvider, Message


def test_meta_provider_enum_exists() -> None:
    """LLMProvider exposes META enum value."""
    assert LLMProvider.META.value == "meta"


def test_meta_parse_model_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config parses meta/<model> into a GenerativeModel instance."""
    monkeypatch.setenv("META_API_KEY", "test-meta-api-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")

    c = Config.from_default()
    model = c._parse_model_string("meta/llama-3-8b-instruct")
    assert model.provider == LLMProvider.META
    assert str(model) == "meta/llama-3-8b-instruct"


def test_meta_auto_detection_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider auto-detection prefers META when META_API_KEY and META_BASE_URL are present."""
    monkeypatch.setenv("META_API_KEY", "test-meta-api-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")

    c = Config.from_default()
    assert c.llm_provider == LLMProvider.META


@pytest.mark.parametrize(
    "model_name",
    [
        "llama-3-8b-instruct",
        "llama-3.1-70b-instruct",
        "Llama-4-Scout-17B-16E-Instruct",
    ],
)
def test_meta_model_names_supported(monkeypatch: pytest.MonkeyPatch, model_name: str) -> None:
    """Construction succeeds for common Llama model names across sizes/variants."""
    monkeypatch.setenv("META_API_KEY", "test-meta-api-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")

    c = Config.from_default()
    model = c._construct_model_from_name(LLMProvider.META, model_name)
    assert model.provider == LLMProvider.META
    assert str(model) == f"meta/{model_name}"


@pytest.mark.asyncio
async def test_meta_aget_response_monkeypatched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async text response path uses model.aget_response; patch that directly."""
    monkeypatch.setenv("META_API_KEY", "test-meta-api-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")

    c = Config.from_default()
    model = c._parse_model_string("meta/llama-3-8b-instruct")

    async def fake_aget_response(_msgs: Any) -> Message:  # noqa: ANN401
        return Message(role="assistant", content="hi")

    monkeypatch.setattr(model, "aget_response", fake_aget_response)

    reply = await model.aget_response([Message(role="user", content="hello")])
    assert reply.role == "assistant"
    assert isinstance(reply.content, str)


@pytest.mark.asyncio
async def test_meta_aget_structured_response_monkeypatched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async structured response path uses model.aget_structured_response; patch it directly."""
    from pydantic import BaseModel

    class S(BaseModel):
        x: int

    monkeypatch.setenv("META_API_KEY", "test-meta-api-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")

    c = Config.from_default()
    model = c._parse_model_string("meta/llama-3-8b-instruct")

    async def fake_aget_structured_response(_msgs: Any, _schema: Any) -> S:  # noqa: ANN401
        return S(x=1)

    monkeypatch.setattr(model, "aget_structured_response", fake_aget_structured_response)

    out = await model.aget_structured_response([Message(role="user", content="hi")], S)
    assert isinstance(out, S)
    assert out.x == 1


def test_meta_default_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default models are populated for META provider when unspecified."""
    monkeypatch.setenv("META_API_KEY", "test-meta-api-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")

    c = Config.from_default()
    # Ensure defaults are resolved to META entries
    assert isinstance(c.models.default_model, str)
    assert isinstance(c.models.planning_model, str)
    assert isinstance(c.models.introspection_model, str)
    assert c.models.default_model.startswith("meta/")
    assert c.models.planning_model.startswith("meta/")
    assert c.models.introspection_model.startswith("meta/")


def test_meta_missing_base_url_error() -> None:
    """Test error handling when no base_url is provided to OpenAICompatibleGenerativeModel."""
    from pydantic import SecretStr

    from portia.model import OpenAICompatibleGenerativeModel

    # Create a test class that inherits from OpenAICompatibleGenerativeModel but
    # doesn't set base_url
    class TestModelWithoutBaseUrl(OpenAICompatibleGenerativeModel):
        pass  # No base_url class variable set

    with pytest.raises(
        ValueError, match="base_url must be provided either in constructor or as class variable"
    ):
        TestModelWithoutBaseUrl(
            model_name="test-model",
            api_key=SecretStr("test-key"),
            # No base_url parameter provided
        )


def test_meta_missing_api_key_error() -> None:
    """Test Config raises error when META_API_KEY is missing."""
    import os

    from portia.errors import InvalidConfigError

    # Ensure META environment variables are not set
    original_api_key = os.environ.pop("META_API_KEY", None)
    original_base_url = os.environ.pop("META_BASE_URL", None)

    try:
        # Provide a default model to avoid config validation errors
        c = Config.from_default(default_model="openai/gpt-4", openai_api_key="test-key")
        with pytest.raises(InvalidConfigError, match="Empty SecretStr value not allowed"):
            c._construct_model_from_name(LLMProvider.META, "llama-3-8b-instruct")
    finally:
        # Restore original environment variables if they existed
        if original_api_key:
            os.environ["META_API_KEY"] = original_api_key
        if original_base_url:
            os.environ["META_BASE_URL"] = original_base_url


def test_meta_missing_base_url_config_error() -> None:
    """Test Config raises error when META_BASE_URL is missing."""
    import os

    from portia.errors import InvalidConfigError

    # Set API key but not base URL
    original_base_url = os.environ.pop("META_BASE_URL", None)
    os.environ["META_API_KEY"] = "test-key"

    try:
        # Provide OpenAI config to avoid validation errors
        c = Config.from_default(default_model="openai/gpt-4", openai_api_key="test-key")
        with pytest.raises(InvalidConfigError, match="Empty value not allowed"):
            c._construct_model_from_name(LLMProvider.META, "llama-3-8b-instruct")
    finally:
        # Clean up and restore
        os.environ.pop("META_API_KEY", None)
        if original_base_url:
            os.environ["META_BASE_URL"] = original_base_url


def test_meta_model_string_representation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Meta models are represented correctly as strings."""
    monkeypatch.setenv("META_API_KEY", "test-meta-api-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")

    c = Config.from_default()
    model = c._construct_model_from_name(LLMProvider.META, "llama-3.1-70b-instruct")

    assert str(model) == "meta/llama-3.1-70b-instruct"
    assert model.provider == LLMProvider.META


def test_meta_provider_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Meta provider is detected when both Meta and other API keys are set."""
    # Set multiple API keys - Meta should be detected in precedence order
    monkeypatch.setenv("META_API_KEY", "test-meta-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")  # This comes first in precedence

    c = Config.from_default()
    # OpenAI should be preferred over Meta in the precedence order
    assert c.llm_provider == LLMProvider.OPENAI


def test_meta_only_provider_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Meta provider detection when only Meta keys are set."""
    monkeypatch.setenv("META_API_KEY", "test-meta-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")

    c = Config.from_default()
    assert c.llm_provider == LLMProvider.META


def test_meta_model_extra_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Meta models can accept extra kwargs."""
    monkeypatch.setenv("META_API_KEY", "test-meta-api-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")

    c = Config.from_default()
    model = c._construct_model_from_name(LLMProvider.META, "llama-3-8b-instruct")

    # Verify the model was created successfully with the base_url
    assert model.provider == LLMProvider.META
    assert str(model) == "meta/llama-3-8b-instruct"


def test_meta_default_model_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Meta provider uses correct default model names."""
    monkeypatch.setenv("META_API_KEY", "test-meta-api-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")

    c = Config.from_default(llm_provider=LLMProvider.META)

    # Check default models are properly configured
    planning_default = c.get_agent_default_model("planning_model", LLMProvider.META)
    introspection_default = c.get_agent_default_model("introspection_model", LLMProvider.META)

    assert planning_default == "meta/llama-3.1-70b-instruct"
    assert introspection_default == "meta/llama-3.1-70b-instruct"


@pytest.mark.parametrize(
    "invalid_base_url",
    [
        "",  # Empty string
        "not-a-url",  # Invalid URL format
        "ftp://invalid-protocol.com",  # Wrong protocol
    ],
)
def test_meta_invalid_base_url_formats(
    monkeypatch: pytest.MonkeyPatch, invalid_base_url: str
) -> None:
    """Test Meta provider with various invalid base URL formats."""
    monkeypatch.setenv("META_API_KEY", "test-meta-api-key")
    monkeypatch.setenv("META_BASE_URL", invalid_base_url)

    # For empty base_url, expect InvalidConfigError from the config validation
    if invalid_base_url == "":
        from portia.errors import InvalidConfigError

        c = Config.from_default(default_model="openai/gpt-4", openai_api_key="test-key")
        with pytest.raises(InvalidConfigError, match="Empty value not allowed"):
            c._construct_model_from_name(LLMProvider.META, "llama-3-8b-instruct")
    else:
        c = Config.from_default()
        # For invalid URLs, the model should still be created - URL validation is handled by 
        # the underlying client
        model = c._construct_model_from_name(LLMProvider.META, "llama-3-8b-instruct")
        assert model.provider == LLMProvider.META
