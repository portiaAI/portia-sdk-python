"""Tests for portia classes."""

import pytest
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
)
from portia.errors import ConfigNotFoundError, InvalidConfigError
from portia.model import (
    AnthropicGenerativeModel,
    AzureOpenAIGenerativeModel,
    GenerativeModel,
    GoogleGenAiGenerativeModel,
    LLMProvider,
    MistralAIGenerativeModel,
    OpenAIGenerativeModel,
)

PROVIDER_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MISTRAL_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "OLLAMA_BASE_URL",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the provider env vars."""
    for env_var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


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
    c = Config.from_default(storage_class="MEMORY")
    assert c.storage_class == StorageClass.MEMORY

    c = Config.from_default(storage_class="DISK", storage_dir="/test")
    assert c.storage_class == StorageClass.DISK
    assert c.storage_dir == "/test"

    # Need to specify storage_dir if using DISK
    with pytest.raises(InvalidConfigError):
        c = Config.from_default(storage_class="DISK")

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


@pytest.mark.parametrize(
    ("model_string", "model_type", "present_env_vars"),
    [
        ("openai/o1-preview", OpenAIGenerativeModel, ["OPENAI_API_KEY"]),
        ("anthropic/claude-3-5-haiku-latest", AnthropicGenerativeModel, ["ANTHROPIC_API_KEY"]),
        ("mistralai/mistral-tiny-latest", MistralAIGenerativeModel, ["MISTRAL_API_KEY"]),
        ("google/gemini-2.5-preview", GoogleGenAiGenerativeModel, ["GOOGLE_API_KEY"]),
        (
            "azure-openai/gpt-4",
            AzureOpenAIGenerativeModel,
            ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
        ),
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
            extest_set_agent_model_default_model_not_settest_set_agent_model_default_model_not_setecution_agent_type=ExecutionAgentType.DEFAULT,
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
    with pytest.raises(InvalidConfigError, match="All models must be instantiable"):
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
    with pytest.raises(InvalidConfigError, match="All models must be instantiable"):
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
        ({"GOOGLE_API_KEY": "test-google-api-key"}, LLMProvider.GOOGLE_GENERATIVE_AI),
        (
            {
                "AZURE_OPENAI_API_KEY": "test-azure-openai-api-key",
                "AZURE_OPENAI_ENDPOINT": "test-azure-openai-endpoint",
            },
            LLMProvider.AZURE_OPENAI,
        ),
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
        ({"google_api_key": "test-google-api-key"}, LLMProvider.GOOGLE_GENERATIVE_AI),
        (
            {
                "azure_openai_api_key": "test-azure-openai-api-key",
                "azure_openai_endpoint": "test-azure-openai-endpoint",
            },
            LLMProvider.AZURE_OPENAI,
        ),
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
    assert planning_model.reasoning_enabled is True

    introspection_model = c.get_introspection_model()
    assert isinstance(introspection_model, AnthropicGenerativeModel)
    assert introspection_model.model_name == "claude-3-7-sonnet-latest"
    assert introspection_model.reasoning_enabled is True

    default_model = c.get_default_model()
    assert isinstance(default_model, AnthropicGenerativeModel)
    assert default_model.model_name == "claude-3-7-sonnet-latest"
    assert default_model.reasoning_enabled is False


def test_provider_default_models_with_reasoning_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that OpenAI models with reasoning in PROVIDER_DEFAULT_MODELS work correctly."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    c = Config.from_default(llm_provider=LLMProvider.OPENAI)

    planning_model = c.get_planning_model()
    assert isinstance(planning_model, OpenAIGenerativeModel)
    assert planning_model.reasoning_enabled is True

    introspection_model = c.get_introspection_model()
    assert isinstance(introspection_model, OpenAIGenerativeModel)
    assert introspection_model.reasoning_enabled is True

    default_model = c.get_default_model()
    assert isinstance(default_model, OpenAIGenerativeModel)
    assert default_model.reasoning_enabled is False
