"""Unit tests for Groq model provider integration."""

import pytest
from pydantic import SecretStr

from portia.config import Config
from portia.model import GroqGenerativeModel, LLMProvider


def test_groq_provider_enum():
    """Test that GROQ is available in LLMProvider enum."""
    assert hasattr(LLMProvider, "GROQ")
    assert LLMProvider.GROQ.value == "groq"


def test_groq_model_instantiation():
    """Test that GroqGenerativeModel can be instantiated with proper parameters."""
    model = GroqGenerativeModel(
        model_name="llama3-8b-8192",
        api_key=SecretStr("test-groq-api-key"),
        temperature=0.0,
    )
    
    assert model.provider == LLMProvider.GROQ
    assert model.model_name == "llama3-8b-8192"
    # Verify the base URL is set correctly for Groq (checking openai_api_base)
    assert "groq.com" in str(model._client.openai_api_base)


def test_groq_config_integration():
    """Test that Config can be created with Groq provider and API key."""
    config = Config(
        llm_provider=LLMProvider.GROQ,
        groq_api_key=SecretStr("test-groq-api-key"),
        default_model="groq/llama3-8b-8192",
        planning_model="groq/llama3-70b-8192",
        execution_model="groq/llama3-8b-8192",
        introspection_model="groq/llama3-8b-8192",
    )
    
    assert config.llm_provider == LLMProvider.GROQ
    assert config.groq_api_key.get_secret_value() == "test-groq-api-key"
    assert config.models.default_model == "groq/llama3-8b-8192"


def test_groq_auto_detection(monkeypatch: pytest.MonkeyPatch):
    """Test that Groq provider is auto-detected when GROQ_API_KEY is set."""
    from portia.config import llm_provider_default_from_api_keys
    
    # Set GROQ_API_KEY in environment
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-api-key")
    
    # Should auto-detect Groq
    detected_provider = llm_provider_default_from_api_keys()
    assert detected_provider == LLMProvider.GROQ


def test_groq_model_parsing():
    """Test that groq/<model> strings are parsed correctly."""
    config = Config(
        llm_provider=LLMProvider.GROQ,
        groq_api_key=SecretStr("test-groq-api-key"),
        default_model="groq/llama3-8b-8192",
    )
    
    model = config._parse_model_string("groq/llama3-8b-8192")
    assert isinstance(model, GroqGenerativeModel)
    assert model.provider == LLMProvider.GROQ
    assert model.model_name == "llama3-8b-8192"


@pytest.mark.parametrize("model_name", [
    "llama3-8b-8192",
    "llama3-70b-8192", 
    "mixtral-8x7b-32768",
    "gemma-7b-it",
])
def test_groq_model_names(model_name: str):
    """Test various Groq model names work correctly."""
    model = GroqGenerativeModel(
        model_name=model_name,
        api_key=SecretStr("test-groq-api-key"),
    )
    
    assert model.model_name == model_name
    assert model.provider == LLMProvider.GROQ


def test_groq_default_models():
    """Test that Groq has appropriate default models configured."""
    from portia.config import Config
    
    config = Config(
        llm_provider=LLMProvider.GROQ,
        groq_api_key=SecretStr("test-groq-api-key"),
    )
    
    # Test default model assignment
    default_model = config.get_agent_default_model("default_model", LLMProvider.GROQ)
    assert default_model == "groq/llama3-8b-8192"
    
    planning_model = config.get_agent_default_model("planning_model", LLMProvider.GROQ)
    assert planning_model == "groq/llama3-70b-8192"
    
    introspection_model = config.get_agent_default_model("introspection_model", LLMProvider.GROQ)
    assert introspection_model == "groq/llama3-8b-8192" 