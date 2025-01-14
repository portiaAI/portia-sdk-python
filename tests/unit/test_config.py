"""Tests for runner classes."""

import tempfile
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from portia.config import (
    AgentType,
    Config,
    LLMModel,
    LLMProvider,
    LogLevel,
    StorageClass,
)
from portia.errors import ConfigNotFoundError, InvalidConfigError


def test_runner_config_from_file() -> None:
    """Test loading configuration from a file."""
    config_data = """{
"portia_api_key": "file-key",
"openai_api_key": "file-openai-key",
"llm_model_temperature": 10,
"storage_class": "MEMORY",
"llm_provider": "OPENAI",
"llm_model_name": "gpt-4o-mini",
"llm_model_seed": 443,
"default_agent_type": "VERIFIER"
}"""

    with tempfile.NamedTemporaryFile("w", delete=True, suffix=".json") as temp_file:
        temp_file.write(config_data)
        temp_file.flush()

        config_file = Path(temp_file.name)

        config = Config.from_file(config_file)

        assert config.must_get_raw_api_key("portia_api_key") == "file-key"
        assert config.must_get_raw_api_key("openai_api_key") == "file-openai-key"
        assert config.default_agent_type == AgentType.VERIFIER
        assert config.llm_model_temperature == 10


def test_from_default() -> None:
    """Test from default."""
    c = Config.from_default(
        default_log_level=LogLevel.CRITICAL,
        openai_api_key=SecretStr("123"),
    )
    assert c.default_log_level == LogLevel.CRITICAL


def test_set_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test setting keys."""
    monkeypatch.setenv("PORTIA_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    c = Config.from_default(default_log_level=LogLevel.CRITICAL)
    assert c.portia_api_key == SecretStr("test-key")
    assert c.openai_api_key == SecretStr("test-openai-key")
    assert c.anthropic_api_key == SecretStr("test-anthropic-key")
    assert c.mistralai_api_key == SecretStr("test-mistral-key")


def test_getters() -> None:
    """Test getters work."""
    c = Config.from_default(
        openai_api_key=SecretStr("123"),
    )

    assert c.has_api_key("openai_api_key")

    with pytest.raises(ConfigNotFoundError):
        c.must_get("not real", str)

    c = Config.from_default(
        openai_api_key=SecretStr("123"),
        portia_api_key=SecretStr(""),
        portia_api_endpoint="",
    )
    with pytest.raises(InvalidConfigError):
        c.must_get("portia_api_key", int)

    with pytest.raises(InvalidConfigError):
        c.must_get_raw_api_key("portia_api_key")

    with pytest.raises(InvalidConfigError):
        c.must_get("portia_api_endpoint", str)

    # mismatch between provider and model
    with pytest.raises(InvalidConfigError):
        Config(
            storage_class=StorageClass.MEMORY,
            llm_provider=LLMProvider.OPENAI,
            llm_model_name=LLMModel.CLAUDE_3_OPUS_LATEST,
            llm_model_temperature=0,
            llm_model_seed=443,
            default_agent_type=AgentType.VERIFIER,
        )

    # no api key for provider model
    for provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.MISTRALAI]:
        with pytest.raises(InvalidConfigError):
            Config(
                storage_class=StorageClass.MEMORY,
                llm_provider=provider,
                openai_api_key=SecretStr(""),
                llm_model_name=LLMModel.GPT_4_O_MINI,
                llm_model_temperature=0,
                llm_model_seed=443,
                default_agent_type=AgentType.VERIFIER,
            )

    # negative temperature
    with pytest.raises(ValidationError):
        Config(
            storage_class=StorageClass.MEMORY,
            llm_provider=LLMProvider.OPENAI,
            llm_model_name=LLMModel.GPT_4_O_MINI,
            llm_model_temperature=0,
            llm_model_seed=-443,
            default_agent_type=AgentType.VERIFIER,
        )
    # no Portia API KEy
    with pytest.raises(InvalidConfigError):
        Config(
            storage_class=StorageClass.CLOUD,
            portia_api_key=SecretStr(""),
            llm_provider=LLMProvider.OPENAI,
            llm_model_name=LLMModel.GPT_4_O_MINI,
            llm_model_temperature=0,
            llm_model_seed=443,
            default_agent_type=AgentType.VERIFIER,
        )


def test_get_default_model() -> None:
    """Test getting default model."""
    assert LLMProvider.OPENAI.default_model() == LLMModel.GPT_4_O_MINI
    assert LLMProvider.ANTHROPIC.default_model() == LLMModel.CLAUDE_3_5_SONNET
    assert LLMProvider.MISTRALAI.default_model() == LLMModel.MISTRAL_LARGE_LATEST


@pytest.mark.parametrize("model", list(LLMModel))
def test_all_models_have_provider(model: LLMModel) -> None:
    """Test all models have a provider."""
    assert model.provider() is not None


@pytest.mark.parametrize("provider", list(LLMProvider))
def test_all_providers_have_associated_model(provider: LLMProvider) -> None:
    """Test all providers have an associated model."""
    assert provider.associated_models() is not None
