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
