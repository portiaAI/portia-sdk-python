"""Tests for runner classes."""

import tempfile
from pathlib import Path

import pytest
from pydantic import SecretStr

from portia.config import (
    AgentType,
    Config,
    LLMModel,
    LLMProvider,
    LogLevel,
    PlannerType,
    StorageClass,
)
from portia.errors import ConfigNotFoundError, InvalidConfigError


def test_runner_config_from_file() -> None:
    """Test loading configuration from a file."""
    config_data = """{
"portia_api_key": "file-key",
"openai_api_key": "file-openai-key",
"storage_class": "MEMORY",
"planning_llm_model_name": "GPT_4_O_MINI",
"execution_llm_model_name": "GPT_4_O_MINI",
"llm_tool_model_name": "GPT_4_O_MINI",
"summariser_llm_model_name": "GPT_4_O_MINI",
"default_agent_type": "VERIFIER",
"default_planner": "ONE_SHOT"
}"""

    with tempfile.NamedTemporaryFile("w", delete=True, suffix=".json") as temp_file:
        temp_file.write(config_data)
        temp_file.flush()

        config_file = Path(temp_file.name)

        config = Config.from_file(config_file)

        assert config.must_get_raw_api_key("portia_api_key") == "file-key"
        assert config.must_get_raw_api_key("openai_api_key") == "file-openai-key"
        assert config.default_agent_type == AgentType.VERIFIER
        assert config.planning_llm_model_name == LLMModel.GPT_4_O_MINI
        assert config.execution_llm_model_name == LLMModel.GPT_4_O_MINI
        assert config.llm_tool_model_name == LLMModel.GPT_4_O_MINI
        assert config.summariser_llm_model_name == LLMModel.GPT_4_O_MINI


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


def test_set_with_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test setting keys as string."""
    monkeypatch.setenv("PORTIA_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    # storage
    c = Config.from_default(storage_class="DISK")
    assert c.storage_class == StorageClass.DISK
    with pytest.raises(InvalidConfigError):
        c = Config.from_default(storage_class="OTHER")

    with pytest.raises(InvalidConfigError):
        c = Config.from_default(storage_class=123)

    # log level
    c = Config.from_default(default_log_level="CRITICAL")
    assert c.default_log_level == LogLevel.CRITICAL
    with pytest.raises(InvalidConfigError):
        c = Config.from_default(default_log_level="some level")

    # LLM provider + model

    # default_agent_type
    c = Config.from_default(default_agent_type="verifier")
    assert c.default_agent_type == AgentType.VERIFIER
    with pytest.raises(InvalidConfigError):
        c = Config.from_default(default_agent_type="my agent")


def test_set_llms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test setting LLM models."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")

    # Models can be set individually
    c = Config.from_default(
        planning_llm_model_name=LLMModel.GPT_4_O,
        execution_llm_model_name=LLMModel.GPT_4_O_MINI,
        llm_tool_model_name=LLMModel.CLAUDE_3_OPUS,
        summariser_llm_model_name=LLMModel.CLAUDE_3_5_HAIKU,
    )
    assert c.planning_llm_model_name == LLMModel.GPT_4_O
    assert c.execution_llm_model_name == LLMModel.GPT_4_O_MINI
    assert c.llm_tool_model_name == LLMModel.CLAUDE_3_OPUS
    assert c.summariser_llm_model_name == LLMModel.CLAUDE_3_5_HAIKU

    # llm_model_name sets all models
    c = Config.from_default(llm_model_name="mistral_large_latest")
    assert c.planning_llm_model_name == LLMModel.MISTRAL_LARGE_LATEST
    assert c.execution_llm_model_name == LLMModel.MISTRAL_LARGE_LATEST
    assert c.llm_tool_model_name == LLMModel.MISTRAL_LARGE_LATEST
    assert c.summariser_llm_model_name == LLMModel.MISTRAL_LARGE_LATEST

    # llm_provider sets default model for all providers
    c = Config.from_default(llm_provider="mistralai")
    assert c.planning_llm_model_name == LLMModel.MISTRAL_LARGE_LATEST
    assert c.execution_llm_model_name == LLMModel.MISTRAL_LARGE_LATEST
    assert c.llm_tool_model_name == LLMModel.MISTRAL_LARGE_LATEST
    assert c.summariser_llm_model_name == LLMModel.MISTRAL_LARGE_LATEST

    # With nothing specified, it chooses a model we have API keys for
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    c = Config.from_default()
    assert c.planning_llm_model_name == LLMModel.MISTRAL_LARGE_LATEST
    assert c.execution_llm_model_name == LLMModel.MISTRAL_LARGE_LATEST
    assert c.llm_tool_model_name == LLMModel.MISTRAL_LARGE_LATEST
    assert c.summariser_llm_model_name == LLMModel.MISTRAL_LARGE_LATEST

    # With all API key set, correct default models are chosen
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    c = Config.from_default()
    assert c.planning_llm_model_name == LLMModel.O3_MINI
    assert c.execution_llm_model_name == LLMModel.GPT_4_O
    assert c.llm_tool_model_name == LLMModel.GPT_4_O
    assert c.summariser_llm_model_name == LLMModel.GPT_4_O

    # With all API key set, correct default models are chosen
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    c = Config.from_default()
    assert c.planning_llm_model_name == LLMModel.CLAUDE_3_5_SONNET
    assert c.execution_llm_model_name == LLMModel.CLAUDE_3_5_SONNET
    assert c.llm_tool_model_name == LLMModel.CLAUDE_3_5_SONNET
    assert c.summariser_llm_model_name == LLMModel.CLAUDE_3_5_SONNET

    # Mismatch between provider and model
    with pytest.raises(InvalidConfigError):
        Config.from_default(
            storage_class=StorageClass.MEMORY,
            llm_provider=LLMProvider.OPENAI,
            llm_model_name=LLMModel.CLAUDE_3_OPUS,
            default_agent_type=AgentType.VERIFIER,
            default_planner=PlannerType.ONE_SHOT,
        )

    # No api key for provider model
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    for provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.MISTRALAI]:
        with pytest.raises(InvalidConfigError):
            Config.from_default(
                storage_class=StorageClass.MEMORY,
                llm_provider=provider,
                default_agent_type=AgentType.VERIFIER,
                default_planner=PlannerType.ONE_SHOT,
            )

    # Unrecognised providers error
    with pytest.raises(InvalidConfigError):
        c = Config.from_default(llm_provider="personal", llm_model_name="other-model")


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

    # no Portia API Key
    with pytest.raises(InvalidConfigError):
        Config.from_default(
            storage_class=StorageClass.CLOUD,
            portia_api_key=SecretStr(""),
            default_agent_type=AgentType.VERIFIER,
            default_planner=PlannerType.ONE_SHOT,
        )


@pytest.mark.parametrize("model", list(LLMModel))
def test_all_models_have_provider(model: LLMModel) -> None:
    """Test all models have a provider."""
    assert model.provider() is not None


@pytest.mark.parametrize("provider", list(LLMProvider))
def test_all_providers_have_associated_model(provider: LLMProvider) -> None:
    """Test all providers have an associated model."""
    assert provider.associated_models() is not None
