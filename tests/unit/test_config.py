"""Tests for portia classes."""

import os
import secrets
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import toml
from langchain_core.caches import InMemoryCache
from pydantic import SecretStr

from portia.config import (
    FEATURE_FLAG_AGENT_MEMORY_ENABLED,
    SUPPORTED_OPENAI_MODELS,
    Config,
    ExecutionAgentType,
    GenerativeModelsConfig,
    LLMModel,
    LogLevel,
    PlanningAgentType,
    StorageClass,
    default_config,
    parse_str_to_enum,
)
from portia.config_loader import ConfigLoader, ensure_config_directory
from portia.errors import ConfigNotFoundError, InvalidConfigError
from portia.model import (
    AmazonBedrockGenerativeModel,
    AnthropicGenerativeModel,
    AzureOpenAIGenerativeModel,
    GenerativeModel,
    GoogleGenAiGenerativeModel,
    LLMProvider,
    MistralAIGenerativeModel,
    OpenAIGenerativeModel,
    OpenRouterGenerativeModel,
    _llm_cache,
)

PROVIDER_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MISTRAL_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "OLLAMA_BASE_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "OPENROUTER_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the provider env vars."""
    for env_var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


def test_from_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test from default."""
    monkeypatch.delenv("LLM_REDIS_CACHE_URL", raising=False)
    c = Config.from_default(
        default_log_level=LogLevel.CRITICAL,
        openai_api_key=SecretStr("123"),
    )
    assert c.default_log_level == LogLevel.CRITICAL
    assert c.execution_agent_type == ExecutionAgentType.ONE_SHOT
    assert c.planning_agent_type == PlanningAgentType.DEFAULT
    if os.getenv("PORTIA_API_KEY"):
        assert c.storage_class == StorageClass.CLOUD
    else:
        assert c.storage_class == StorageClass.MEMORY
    assert c.llm_redis_cache_url is None
    assert _llm_cache.get() is None


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
    c = Config.from_default(storage_class="MEMORY")
    assert c.storage_class == StorageClass.MEMORY
    c = Config.from_default(storage_class="DISK", storage_dir="/test")
    assert c.storage_class == StorageClass.DISK
    assert c.storage_dir == "/test"
    c = Config.from_default(storage_class="DISK")
    assert c.storage_class == StorageClass.DISK
    assert c.storage_dir is None  # Will default to .portia in DiskFileStorage
    with pytest.raises(InvalidConfigError):
        c = Config.from_default(storage_class="OTHER")
    with pytest.raises(InvalidConfigError):
        c = Config.from_default(storage_class=123)
    # log level
    c = Config.from_default(default_log_level="CRITICAL")
    assert c.default_log_level == LogLevel.CRITICAL
    with pytest.raises(InvalidConfigError):
        c = Config.from_default(default_log_level="some level")
    # execution_agent_type
    c = Config.from_default(execution_agent_type="default")
    assert c.execution_agent_type == ExecutionAgentType.DEFAULT
    with pytest.raises(InvalidConfigError):
        c = Config.from_default(execution_agent_type="my agent")
    # Large output threshold value
    c = Config.from_default(
        large_output_threshold_tokens=100,
        feature_flags={
            FEATURE_FLAG_AGENT_MEMORY_ENABLED: True,
        },
    )
    assert c.large_output_threshold_tokens == 100
    assert c.exceeds_output_threshold("Test " * 1000)
    c = Config.from_default(
        large_output_threshold_tokens=100,
        feature_flags={
            FEATURE_FLAG_AGENT_MEMORY_ENABLED: False,
        },
    )
    assert c.large_output_threshold_tokens == 100
    assert not c.exceeds_output_threshold("Test " * 1000)


def test_llm_redis_cache_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """llm_redis_cache_url is read from environment variable."""
    mock_redis_cache_instance = MagicMock()
    mock_redis_cache = MagicMock(return_value=mock_redis_cache_instance)
    monkeypatch.setattr("langchain_redis.RedisCache", mock_redis_cache)
    monkeypatch.setenv("LLM_REDIS_CACHE_URL", "redis://localhost:6379/0")
    config = Config.from_default(openai_api_key=SecretStr("123"))
    assert config.llm_redis_cache_url == "redis://localhost:6379/0"
    assert _llm_cache.get() is mock_redis_cache_instance


def test_llm_redis_cache_url_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    """llm_redis_cache_url can be set via kwargs."""
    mock_redis_cache_instance = InMemoryCache()
    mock_redis_cache = MagicMock(return_value=mock_redis_cache_instance)
    monkeypatch.setattr("langchain_redis.RedisCache", mock_redis_cache)
    config = Config.from_default(
        openai_api_key=SecretStr("123"), llm_redis_cache_url="redis://localhost:6379/0"
    )
    assert config.llm_redis_cache_url == "redis://localhost:6379/0"
    assert _llm_cache.get() is mock_redis_cache_instance


@pytest.mark.parametrize(
    ("model_string", "model_type", "present_env_vars"),
    [
        ("openai/o1-preview", OpenAIGenerativeModel, ["OPENAI_API_KEY"]),
        ("anthropic/claude-3-5-haiku-latest", AnthropicGenerativeModel, ["ANTHROPIC_API_KEY"]),
        ("mistralai/mistral-tiny-latest", MistralAIGenerativeModel, ["MISTRAL_API_KEY"]),
        ("google/gemini-2.5-preview", GoogleGenAiGenerativeModel, ["GOOGLE_API_KEY"]),
        (
            "amazon/anthropic.claude-3-sonnet-v1:0",
            AmazonBedrockGenerativeModel,
            ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"],
        ),
        (
            "azure-openai/gpt-4",
            AzureOpenAIGenerativeModel,
            ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
        ),
        ("openrouter/moonshotai/kimi-k2", OpenRouterGenerativeModel, ["OPENROUTER_API_KEY"]),
    ],
)
def test_set_default_model_from_string(
    model_string: str,
    model_type: type[GenerativeModel],
    present_env_vars: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test setting default model from string."""
    for env_var in present_env_vars:
        monkeypatch.setenv(env_var, "test-key")

    # Default model
    c = Config.from_default(default_model=model_string)
    model = c.get_default_model()
    assert isinstance(model, model_type)
    assert str(model) == model_string

    # Planning_model
    c = Config.from_default(planning_model=model_string)
    model = c.get_planning_model()
    assert isinstance(model, model_type)
    assert str(model) == model_string


def test_set_default_model_from_model_instance() -> None:
    """Test setting default model from model instance without provider set."""
    model = OpenAIGenerativeModel(model_name="gpt-4o", api_key=SecretStr("test-openai-key"))
    c = Config.from_default(default_model=model)
    resolved_model = c.get_default_model()
    assert resolved_model is model

    # Planning_model has not been set, and we dont have a provider set, so this returns the
    # default model
    planner_model = c.get_planning_model()
    assert planner_model is model


MODEL_KEYS = sorted(set(GenerativeModelsConfig.model_fields) - {"default_model"})


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_set_agent_model_default_model_not_set_fails(model_key: str) -> None:
    """Test setting agent_model from model instance without default model or provider set."""
    model = OpenAIGenerativeModel(model_name="gpt-4o", api_key=SecretStr("test-openai-key"))
    with pytest.raises(InvalidConfigError):
        _ = Config.from_default(**{model_key: model})


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_set_agent_model_with_string_api_key_env_var_set(
    monkeypatch: pytest.MonkeyPatch,
    model_key: str,
) -> None:
    """Test setting planning_model with string, with correct API key env var present."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    model_str = "openai/gpt-4o"
    c = Config.from_default(**{model_key: model_str})
    method = getattr(c, f"get_{model_key}")
    resolved_model = method()
    assert str(resolved_model) == model_str

    # Provider inferred from env var to be OpenAI, so default model is OpenAI default model
    default_model = c.get_default_model()
    assert isinstance(default_model, OpenAIGenerativeModel)


def test_set_model_with_string_api_key_env_var_not_set() -> None:
    """Test setting planning_model with string, with correct API key env var not present."""
    model_str = "openai/gpt-4o"
    with pytest.raises((ConfigNotFoundError, InvalidConfigError)):
        _ = Config.from_default(default_model=model_str)


def test_set_model_with_string_other_provider_api_key_env_var_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test setting default model from string with no API key env var set.

    In this case, the env var is present for Anthropic, but user sets a Mistral model as
    default_model.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    with pytest.raises((ConfigNotFoundError, InvalidConfigError)):
        _ = Config.from_default(
            default_model="mistralai/mistral-tiny-latest",
            llm_provider="anthropic",
        )


def test_set_default_model_from_string_with_alternative_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test setting model from string from a different provider to what is explicitly set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    c = Config.from_default(default_model="mistralai/mistral-tiny-latest", llm_provider="anthropic")
    model = c.get_default_model()
    assert isinstance(model, MistralAIGenerativeModel)
    assert str(model) == "mistralai/mistral-tiny-latest"

    model = c.get_planning_model()
    assert isinstance(model, AnthropicGenerativeModel)


def test_provider_set_from_env_planner_model_overriden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test when provider is set from an environment variable, and planning_model overriden.

    The planning_model should respect the explicit planning_model, but the default model should
    respect the provider set from the environment variable.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    c = Config.from_default(
        planning_model=AzureOpenAIGenerativeModel(
            model_name="gpt-4o",
            api_key=SecretStr("test-azure-openai-key"),
            azure_endpoint="test-azure-openai-endpoint",
        ),
    )
    model = c.get_planning_model()
    assert isinstance(model, AzureOpenAIGenerativeModel)
    assert str(model) == "azure-openai/gpt-4o"

    default_model = c.get_default_model()
    assert isinstance(default_model, AnthropicGenerativeModel)


def test_set_default_model_and_planning_model_alternative_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test setting default model and planning_model from string with alternative provider."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    c = Config.from_default(
        default_model="mistralai/mistral-tiny-latest",
        planning_model="google/gemini-1.5-flash",
        llm_provider="anthropic",
    )
    model = c.get_default_model()
    assert isinstance(model, MistralAIGenerativeModel)
    assert str(model) == "mistralai/mistral-tiny-latest"

    model = c.get_planning_model()
    assert isinstance(model, GoogleGenAiGenerativeModel)
    assert str(model) == "google/gemini-1.5-flash"


def test_set_default_model_alternative_provider_missing_api_key_explicit_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test setting default model with a model instance different to LLM provider.

    The user sets the Mistral model object explicitly. This works, because the API key is
    set in the constructor of GenerativeModel.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    config = Config.from_default(
        default_model=MistralAIGenerativeModel(
            model_name="mistral-tiny-latest",
            api_key=SecretStr("test-mistral-key"),
        ),
        llm_provider="anthropic",
    )
    assert isinstance(config.get_default_model(), MistralAIGenerativeModel)
    assert str(config.get_default_model()) == "mistralai/mistral-tiny-latest"


def test_set_default_and_planner_model_with_instances_no_provider_set() -> None:
    """Test setting default model and planning_model with model instances, and no provider set."""
    config = Config.from_default(
        default_model=MistralAIGenerativeModel(
            model_name="mistral-tiny-latest",
            api_key=SecretStr("test-mistral-key"),
        ),
        planning_model=OpenAIGenerativeModel(
            model_name="gpt-4o",
            api_key=SecretStr("test-openai-key"),
        ),
    )
    assert isinstance(config.get_default_model(), MistralAIGenerativeModel)
    assert str(config.get_default_model()) == "mistralai/mistral-tiny-latest"
    assert isinstance(config.get_planning_model(), OpenAIGenerativeModel)
    assert str(config.get_planning_model()) == "openai/gpt-4o"


def test_get_planning_model_azure() -> None:
    """Test resolve model for Azure OpenAI."""
    c = Config.from_default(
        llm_provider=LLMProvider.AZURE_OPENAI,
        azure_openai_endpoint="http://test-azure-openai-endpoint",
        azure_openai_api_key="test-azure-openai-api-key",
    )
    assert isinstance(c.get_planning_model(), AzureOpenAIGenerativeModel)


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
        portia_api_key=SecretStr("123"),
        anthropic_api_key=SecretStr(""),
        portia_api_endpoint="",
        portia_dashboard_url="",
    )
    with pytest.raises(InvalidConfigError):
        c.must_get("portia_api_key", int)

    with pytest.raises(InvalidConfigError):
        c.must_get("portia_api_endpoint", str)

    with pytest.raises(InvalidConfigError):
        c.must_get("portia_dashboard_url", str)

    # no Portia API Key
    with pytest.raises(
        InvalidConfigError,
        match="A Portia API key must be provided if using cloud storage",
    ):
        Config.from_default(
            storage_class=StorageClass.CLOUD,
            portia_api_key=SecretStr(""),
            execution_agent_type=ExecutionAgentType.DEFAULT,
            planning_agent_type=PlanningAgentType.DEFAULT,
            llm_provider=LLMProvider.OPENAI,
            openai_api_key=SecretStr("test-openai-api-key"),
        )


def test_azure_openai_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Azure OpenAI requires endpoint."""
    # Passing both endpoint and api key as kwargs works
    c = Config.from_default(
        llm_provider=LLMProvider.AZURE_OPENAI,
        azure_openai_endpoint="test-azure-openai-endpoint",
        azure_openai_api_key="test-azure-openai-api-key",
    )
    assert c.llm_provider == LLMProvider.AZURE_OPENAI

    # Without endpoint set via kwargs, it errors
    with pytest.raises((ConfigNotFoundError, InvalidConfigError)):
        _ = Config.from_default(
            llm_provider=LLMProvider.AZURE_OPENAI,
            azure_openai_api_key="test-azure-openai-api-key",
        )

    # Without endpoint set via env var, it errors
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-openai-key")
    with pytest.raises((ConfigNotFoundError, InvalidConfigError)):
        Config.from_default(llm_provider=LLMProvider.AZURE_OPENAI)

    # With endpoint set via env var, it works
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "test-azure-openai-endpoint")
    c = Config.from_default(llm_provider=LLMProvider.AZURE_OPENAI)
    assert c.llm_provider == LLMProvider.AZURE_OPENAI


def test_custom_model_from_string_raises_error() -> None:
    """Test custom model from string raises an error."""
    with pytest.raises(ValueError, match="Cannot construct a custom model from a string test"):
        _ = Config.from_default(default_model="custom/test")


def test_set_model_from_llm_model_raises_deprecation_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test setting model from LLMModel raises a warning."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-api-key")
    with pytest.warns(DeprecationWarning):
        c = Config.from_default(
            default_model=LLMModel("gpt-4o"),
            planning_model=LLMModel("openai/o3-mini"),
        )
    assert c.models.default_model == "openai/gpt-4o"
    assert c.models.planning_model == "openai/o3-mini"


def test_check_model_supported_raises_deprecation_warning() -> None:
    """Test checking if a model is supported via SUPPORTED_* models lists raise a warning."""
    with pytest.warns(DeprecationWarning):
        assert "gpt-4o" in SUPPORTED_OPENAI_MODELS


def test_summarizer_model_not_instantiable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test summarizer model is not instantiable."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-api-key")
    with pytest.raises(
        InvalidConfigError,
        match="SUMMARIZER_MODEL is not valid - The value mistralai/mistral-large-latest",
    ):
        Config.from_default(
            default_model="openai/gpt-4o",
            summarizer_model="mistralai/mistral-large-latest",
        )


def test_config_error_resolve_model_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Config.model raises a ConfigModelResolutionError if no model is found."""
    # Setup: Its actually not currently possible to get a model in this state, but
    # it is tested here because Config validation can change independently and we want
    # to make sure the model getters raise a config error if that happens.
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-api-key")
    config = Config.from_default()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config.models = GenerativeModelsConfig()
    config.llm_provider = None

    with pytest.raises(InvalidConfigError):
        config.get_default_model()

    with pytest.raises(InvalidConfigError):
        config.get_planning_model()


def test_config_model_in_kwargs_and_models_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Config.model in kwargs and models raises an InvalidConfigError."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-api-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-api-key")
    with pytest.raises(InvalidConfigError):
        Config.from_default(
            default_model="openai/gpt-4o",
            models={"default_model": "mistralai/mistral-tiny-latest"},
        )


def test_no_provider_or_default_model_raises_error() -> None:
    """Test no provider or default model raises an InvalidConfigError."""
    with pytest.raises(
        InvalidConfigError,
        match=".*Either llm_provider must be set, default model must be set, or an API key must be "
        "provided to allow for automatic model selection*",
    ):
        Config.from_default()


def test_llm_model_name_deprecation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test using llm_model_name raises a DeprecationWarning (but works)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-api-key")
    with pytest.warns(DeprecationWarning):
        c = Config.from_default(llm_model_name="openai/gpt-4o")
    assert c.models.default_model == "openai/gpt-4o"


@pytest.mark.parametrize(
    ("legacy_model_key", "new_model_key"),
    [
        ("planning_model_name", "planning_model"),
        ("execution_model_name", "execution_model"),
        ("introspection_model_name", "introspection_model"),
        ("summariser_model_name", "summarizer_model"),
        ("default_model_name", "default_model"),
    ],
)
def test_legacy_model_kwargs_deprecation(
    legacy_model_key: str,
    new_model_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test legacy model kwargs raise a DeprecationWarning (but works)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-api-key")
    with pytest.warns(DeprecationWarning):
        c = Config.from_default(
            **{legacy_model_key: "openai/o1"},
            llm_provider=LLMProvider.OPENAI,
        )
    assert getattr(c.models, new_model_key) == "openai/o1"


def test_get_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_model for different arg types."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-api-key")
    c = Config.from_default()
    model = c.get_generative_model("openai/gpt-4o")
    assert isinstance(model, OpenAIGenerativeModel)
    assert str(model) == "openai/gpt-4o"

    assert c.get_generative_model(None) is None
    from_instsance = c.get_generative_model(
        OpenAIGenerativeModel(model_name="gpt-4o-mini", api_key=SecretStr("test-openai-api-key")),
    )
    assert isinstance(from_instsance, OpenAIGenerativeModel)
    assert str(from_instsance) == "openai/gpt-4o-mini"


@pytest.mark.parametrize(
    ("env_vars", "provider"),
    [
        ({"OPENAI_API_KEY": "test-openai-api-key"}, LLMProvider.OPENAI),
        ({"ANTHROPIC_API_KEY": "test-anthropic-api-key"}, LLMProvider.ANTHROPIC),
        ({"MISTRAL_API_KEY": "test-mistral-api-key"}, LLMProvider.MISTRALAI),
        ({"GOOGLE_API_KEY": "test-google-api-key"}, LLMProvider.GOOGLE),
        (
            {
                "AWS_ACCESS_KEY_ID": "test-aws-access-key-id",
                "AWS_SECRET_ACCESS_KEY": "test-aws-secret-access-key",
                "AWS_DEFAULT_REGION": "eu-east-1",
            },
            LLMProvider.AMAZON,
        ),
        (
            {
                "AZURE_OPENAI_API_KEY": "test-azure-openai-api-key",
                "AZURE_OPENAI_ENDPOINT": "test-azure-openai-endpoint",
            },
            LLMProvider.AZURE_OPENAI,
        ),
        ({"OPENROUTER_API_KEY": "test-openrouter-api-key"}, LLMProvider.OPENROUTER),
        ({"XAI_API_KEY": "test-xai-api-key"}, LLMProvider.GROK),
    ],
)
def test_llm_provider_default_from_api_keys_env_vars(
    env_vars: dict[str, str],
    provider: LLMProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test LLM provider default from API keys env vars."""
    for env_var_name, env_var_value in env_vars.items():
        monkeypatch.setenv(env_var_name, env_var_value)

    c = Config.from_default()
    assert c.llm_provider == provider


@pytest.mark.parametrize(
    ("config_kwargs", "provider"),
    [
        ({"openai_api_key": "test-openai-api-key"}, LLMProvider.OPENAI),
        ({"anthropic_api_key": "test-anthropic-api-key"}, LLMProvider.ANTHROPIC),
        ({"mistralai_api_key": "test-mistral-api-key"}, LLMProvider.MISTRALAI),
        ({"google_api_key": "test-google-api-key"}, LLMProvider.GOOGLE),
        (
            {
                "aws_access_key_id": "test-aws-access-key-id",
                "aws_secret_access_key": "test-aws-secret-access-key",
                "aws_default_region": "eu-east-1",
            },
            LLMProvider.AMAZON,
        ),
        (
            {
                "azure_openai_api_key": "test-azure-openai-api-key",
                "azure_openai_endpoint": "test-azure-openai-endpoint",
            },
            LLMProvider.AZURE_OPENAI,
        ),
        ({"openrouter_api_key": "test-openrouter-api-key"}, LLMProvider.OPENROUTER),
        ({"grok_api_key": "test-grok-api-key"}, LLMProvider.GROK),
    ],
)
def test_llm_provider_default_from_api_keys_config_kwargs(
    config_kwargs: dict[str, str],
    provider: LLMProvider,
) -> None:
    """Test LLM provider default from API keys config kwargs."""
    c = Config.from_default(**config_kwargs)
    assert c.llm_provider == provider


def test_deprecated_llm_model_cannot_instantiate_from_string() -> None:
    """Test deprecated LLMModel cannot be instantiated from a string."""
    with pytest.raises(ValueError, match="Invalid LLM model"):
        _ = LLMModel("adijabisfbgiwjebr")


def test_provider_default_models_with_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that default models with reasoning in PROVIDER_DEFAULT_MODELS work correctly."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    c = Config.from_default(llm_provider=LLMProvider.ANTHROPIC)

    planning_model = c.get_planning_model()
    assert isinstance(planning_model, AnthropicGenerativeModel)
    assert planning_model.model_name == "claude-3-7-sonnet-latest"
    assert hasattr(planning_model, "_model_kwargs")
    assert "thinking" in planning_model._model_kwargs
    assert planning_model._model_kwargs["thinking"]["type"] == "enabled"

    introspection_model = c.get_introspection_model()
    assert isinstance(introspection_model, AnthropicGenerativeModel)
    assert introspection_model.model_name == "claude-3-7-sonnet-latest"
    assert hasattr(introspection_model, "_model_kwargs")
    assert "thinking" in introspection_model._model_kwargs
    assert introspection_model._model_kwargs["thinking"]["type"] == "enabled"

    default_model = c.get_default_model()
    assert isinstance(default_model, AnthropicGenerativeModel)
    assert default_model.model_name == "claude-3-5-sonnet-latest"
    assert not hasattr(default_model, "_model_kwargs") or "thinking" not in getattr(
        default_model, "_model_kwargs", {}
    )


def test_provider_default_models_with_reasoning_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that OpenAI models with reasoning in PROVIDER_DEFAULT_MODELS work correctly."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    c = Config.from_default(llm_provider=LLMProvider.OPENAI)

    planning_model = c.get_planning_model()
    assert isinstance(planning_model, OpenAIGenerativeModel)
    assert hasattr(planning_model, "_model_kwargs")
    assert "reasoning_effort" in planning_model._model_kwargs
    assert planning_model._model_kwargs["reasoning_effort"] == "medium"

    introspection_model = c.get_introspection_model()
    assert isinstance(introspection_model, OpenAIGenerativeModel)
    assert hasattr(introspection_model, "_model_kwargs")
    assert "reasoning_effort" in introspection_model._model_kwargs
    assert introspection_model._model_kwargs["reasoning_effort"] == "medium"

    default_model = c.get_default_model()
    assert isinstance(default_model, OpenAIGenerativeModel)
    assert not hasattr(default_model, "_model_kwargs") or "reasoning_effort" not in getattr(
        default_model, "_model_kwargs", {}
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("google", LLMProvider.GOOGLE),
        ("google_generative_ai", LLMProvider.GOOGLE),
        ("google-generative-ai", LLMProvider.GOOGLE),
        ("azure-openai", LLMProvider.AZURE_OPENAI),
        ("azure_openai", LLMProvider.AZURE_OPENAI),
        ("anthropic", LLMProvider.ANTHROPIC),
        ("mistralai", LLMProvider.MISTRALAI),
        ("openai", LLMProvider.OPENAI),
        ("amazon", LLMProvider.AMAZON),
        ("grok", LLMProvider.GROK),
    ],
)
def test_parse_str_to_enum(value: str, expected: LLMProvider) -> None:
    """Test parse_str_to_enum works."""
    assert parse_str_to_enum(value, LLMProvider) is expected


def test_fill_default_models_planning_model_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that planning_model is set when None but default_model exists."""
    from portia.config import Config, GenerativeModelsConfig, LLMProvider

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Create config with default_model set but planning_model None
    models = GenerativeModelsConfig(
        default_model="openai/gpt-4.1",
        planning_model=None,  # Explicitly None to trigger the fill logic
    )
    c = Config.from_default(
        llm_provider=LLMProvider.OPENAI, models=models, openai_api_key="test-key"
    )
    # Should set planning_model from provider default
    assert c.models.planning_model == "openai/o3-mini"


def test_fill_default_models_introspection_model_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that introspection_model is set when None but default_model exists."""
    from portia.config import Config, GenerativeModelsConfig, LLMProvider

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    models = GenerativeModelsConfig(default_model="openai/gpt-4.1", introspection_model=None)
    c = Config.from_default(
        llm_provider=LLMProvider.OPENAI, models=models, openai_api_key="test-key"
    )

    assert c.models.introspection_model == "openai/o4-mini"


def test_default_config_all_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test all environment variable overrides in default_config final block."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    monkeypatch.setenv("PORTIA_API_ENDPOINT", "https://api.env.test")
    monkeypatch.setenv("PORTIA_DASHBOARD_URL", "https://dash.env.test")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.env.test")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-aws-key-id")
    aws_secret = secrets.token_hex(16)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", aws_secret)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("AWS_CREDENTIALS_PROFILE_NAME", "my_aws_profile")
    monkeypatch.setenv("LLM_REDIS_CACHE_URL", "redis://localhost:6379/1")

    monkeypatch.setattr("langchain_redis.RedisCache", lambda *_a, **_kw: InMemoryCache())

    cfg = default_config()
    assert cfg.portia_api_endpoint == "https://api.env.test"
    assert cfg.portia_dashboard_url == "https://dash.env.test"
    assert cfg.ollama_base_url == "http://ollama.env.test"
    assert cfg.aws_access_key_id == "test-aws-key-id"
    assert cfg.aws_secret_access_key == aws_secret
    assert cfg.aws_default_region == "us-west-2"
    assert cfg.aws_credentials_profile_name == "my_aws_profile"
    assert cfg.llm_redis_cache_url == "redis://localhost:6379/1"


def make_minimal_config() -> Config:
    """Make a minimal valid Config for testing."""
    return Config(
        llm_provider=LLMProvider.OPENAI,
        openai_api_key=SecretStr("test-key"),
        models=GenerativeModelsConfig(default_model="openai/gpt-4.1"),
    )


def test_must_get_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test must_get_api_key works and raises errors correctly."""
    config = make_minimal_config()
    monkeypatch.setattr(config, "portia_api_key", SecretStr("secret"))
    assert config.must_get_api_key("portia_api_key").get_secret_value() == "secret"


def test_must_get() -> None:
    """Test must_get works and raises errors correctly."""
    config = make_minimal_config()
    assert config.must_get("openai_api_key", SecretStr) == SecretStr("test-key")
    with pytest.raises(ConfigNotFoundError):
        config.must_get("not_present", str)


def test_get_default_model() -> None:
    """Test get_default_model returns a model instance."""
    config = make_minimal_config()
    model = config.get_default_model()
    assert model is not None


def test_env_var_mapping_iterates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that merge_with_env iterates through ENV_VAR_MAPPING and picks up env vars."""
    loader = ConfigLoader()
    monkeypatch.setenv("OPENAI_API_KEY", "env-value")
    config = {}
    merged = loader.merge_with_env(config)
    assert merged["openai_api_key"] == "env-value"


def test_ensure_config_directory_creates_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that ensure_config_directory creates and returns the config dir."""
    monkeypatch.setattr(ConfigLoader, "DEFAULT_CONFIG_DIR", tmp_path / "mycfg")
    config_dir = ensure_config_directory()
    assert config_dir.exists()
    assert config_dir.is_dir()


def test_fill_default_models_both_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that both planning_model and introspection_model are set when None."""
    from portia.config import Config, GenerativeModelsConfig, LLMProvider

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    models = GenerativeModelsConfig(
        default_model="openai/gpt-4.1",
        planning_model=None,
        introspection_model=None,
    )
    c = Config.from_default(
        llm_provider=LLMProvider.OPENAI, models=models, openai_api_key="test-key"
    )
    assert c.models.planning_model == "openai/o3-mini"
    assert c.models.introspection_model == "openai/o4-mini"


def make_openai_config(
    exec_model: str | None = None, default_model: str = "openai/gpt-4o"
) -> Config:
    """Create a Config with OpenAI provider and specified models."""
    models = GenerativeModelsConfig(default_model=default_model, execution_model=exec_model)
    return Config(
        llm_provider=LLMProvider.OPENAI,
        models=models,
        openai_api_key=SecretStr("test-key"),
    )


def test_get_execution_model_returns_specified() -> None:
    """Test get_execution_model returns the specified execution_model."""
    config = make_openai_config(exec_model="openai/gpt-4o")
    model = config.get_execution_model()
    assert isinstance(model, OpenAIGenerativeModel)
    assert str(model) == "openai/gpt-4o"


def test_get_execution_model_falls_back_to_default() -> None:
    """Test get_execution_model falls back to default_model if execution_model is None."""
    config = make_openai_config(exec_model=None, default_model="openai/gpt-4o")
    model = config.get_execution_model()
    assert isinstance(model, OpenAIGenerativeModel)
    assert str(model) == "openai/gpt-4o"


def test_get_generative_model_none_returns_none() -> None:
    """Test get_generative_model returns None if passed None."""
    config = make_openai_config()
    assert config.get_generative_model(None) is None


def test_get_generative_model_string_parses() -> None:
    """Test get_generative_model parses string and returns correct model."""
    config = make_openai_config()
    model = config.get_generative_model("openai/gpt-4o")
    assert isinstance(model, OpenAIGenerativeModel)
    assert str(model) == "openai/gpt-4o"


def test_get_generative_model_instance_returns_instance() -> None:
    """Test get_generative_model returns instance if passed a model instance."""
    config = make_openai_config()
    inst = OpenAIGenerativeModel(model_name="gpt-4o", api_key=SecretStr("test-key"))
    assert config.get_generative_model(inst) is inst


def test_from_local_config_with_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading config from a local config file with a profile."""
    config_file = tmp_path / "config.toml"
    data = {
        "profile": {
            "openai": {
                "llm_provider": "openai",
                "default_model": "openai/gpt-4o",
                "openai_api_key": "test-key",
            }
        }
    }
    with Path.open(config_file, "w") as f:
        toml.dump(data, f)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = Config.from_local_config(profile="openai", config_file=config_file)
    assert config.llm_provider == LLMProvider.OPENAI
    assert config.models.default_model == "openai/gpt-4o"
    assert config.openai_api_key.get_secret_value() == "test-key"


def test_enum_parsing_for_agent_types() -> None:
    """Test lines 814 and 817: execution_agent_type and planning_agent_type.

    string to enum parsing.
    """
    config = Config.from_default(
        execution_agent_type="ONE_SHOT",
        planning_agent_type="DEFAULT",
        portia_api_key="dummy",
        openai_api_key="dummy",
    )
    assert config.execution_agent_type == ExecutionAgentType.ONE_SHOT
    assert config.planning_agent_type == PlanningAgentType.DEFAULT


@patch("portia.config_loader.get_config")
def test_execution_agent_type_parsing(mock_get_config: MagicMock) -> None:
    """Test line 814: execution_agent_type string to enum parsing."""
    mock_get_config.return_value = {
        "llm_provider": "openai",
        "openai_api_key": "dummy",
        "execution_agent_type": "ONE_SHOT",
    }
    config = Config.from_local_config()
    assert config.execution_agent_type == ExecutionAgentType.ONE_SHOT

    mock_get_config.return_value = {
        "llm_provider": "openai",
        "openai_api_key": "dummy",
        "execution_agent_type": "one_shot",
    }
    config = Config.from_local_config()
    assert config.execution_agent_type == ExecutionAgentType.ONE_SHOT

    mock_get_config.return_value = {
        "llm_provider": "openai",
        "openai_api_key": "dummy",
        "execution_agent_type": "one-shot",
    }
    config = Config.from_local_config()
    assert config.execution_agent_type == ExecutionAgentType.ONE_SHOT


@patch("portia.config_loader.get_config")
def test_planning_agent_type_parsing(mock_get_config: MagicMock) -> None:
    """Test line 817: planning_agent_type string to enum parsing."""
    mock_get_config.return_value = {
        "llm_provider": "openai",
        "openai_api_key": "dummy",
        "planning_agent_type": "DEFAULT",
    }
    config = Config.from_local_config()
    assert config.planning_agent_type == PlanningAgentType.DEFAULT

    mock_get_config.return_value = {
        "llm_provider": "openai",
        "openai_api_key": "dummy",
        "planning_agent_type": "default",
    }
    config = Config.from_local_config()
    assert config.planning_agent_type == PlanningAgentType.DEFAULT


@patch("portia.config_loader.get_config")
def test_invalid_execution_agent_type(mock_get_config: MagicMock) -> None:
    """Test line 814: invalid execution_agent_type raises error."""
    mock_get_config.return_value = {"execution_agent_type": "INVALID_TYPE"}
    with pytest.raises(InvalidConfigError):
        Config.from_local_config()


@patch("portia.config_loader.get_config")
def test_invalid_planning_agent_type(mock_get_config: MagicMock) -> None:
    """Test line 817: invalid planning_agent_type raises error."""
    mock_get_config.return_value = {"planning_agent_type": "INVALID_TYPE"}
    with pytest.raises(InvalidConfigError):
        Config.from_local_config()


@patch("portia.config_loader.get_config")
def test_both_together(mock_get_config: MagicMock) -> None:
    """Test both lines 814 and 817 working together."""
    mock_get_config.return_value = {
        "llm_provider": "openai",
        "openai_api_key": "dummy",
        "execution_agent_type": "DEFAULT",
        "planning_agent_type": "DEFAULT",
    }
    config = Config.from_local_config()
    assert config.execution_agent_type == ExecutionAgentType.DEFAULT
    assert config.planning_agent_type == PlanningAgentType.DEFAULT
