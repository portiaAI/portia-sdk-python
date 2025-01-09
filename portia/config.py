"""Configuration for the SDK."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Annotated, TypeVar

from pydantic import AfterValidator, BaseModel, SecretStr, model_validator

from portia.errors import ConfigNotFoundError, InvalidConfigError

T = TypeVar("T")


class StorageClass(Enum):
    """Represent locations plans and workflows are written to."""

    MEMORY = "MEMORY"
    DISK = "DISK"
    CLOUD = "CLOUD"


class LLMProvider(Enum):
    """Enum of LLM providers."""

    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    MISTRALAI = "MISTRALAI"

    def associated_models(self) -> list[LLMModel]:
        """Get the associated models for the provider."""
        match self:
            case LLMProvider.OPENAI:
                return SUPPORTED_OPENAI_MODELS
            case LLMProvider.ANTHROPIC:
                return SUPPORTED_ANTHROPIC_MODELS
            case LLMProvider.MISTRALAI:
                return SUPPORTED_MISTRALAI_MODELS

    def default_model(self) -> LLMModel:
        """Get the default model for the provider."""
        match self:
            case LLMProvider.OPENAI:
                return LLMModel.GPT_4_O_MINI
            case LLMProvider.ANTHROPIC:
                return LLMModel.CLAUDE_3_5_SONNET
            case LLMProvider.MISTRALAI:
                return LLMModel.MISTRAL_LARGE_LATEST


class LLMModel(Enum):
    """Supported Models."""

    # OpenAI
    GPT_4_O = "gpt-4o"
    GPT_4_O_MINI = "gpt-4o-mini"
    GPT_3_5_TURBO = "gpt-3.5-turbo"

    # Anthropic
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-latest"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-latest"
    CLAUDE_3_OPUS_LATEST = "claude-3-opus-latest"

    # MistralAI
    MISTRAL_LARGE_LATEST = "mistral-large-latest"

    def provider(self) -> LLMProvider:
        """Get the associated provider for the model."""
        match self:
            case LLMModel.GPT_4_O | LLMModel.GPT_4_O_MINI | LLMModel.GPT_3_5_TURBO:
                return LLMProvider.OPENAI
            case (
                LLMModel.CLAUDE_3_5_HAIKU
                | LLMModel.CLAUDE_3_5_SONNET
                | LLMModel.CLAUDE_3_OPUS_LATEST
            ):
                return LLMProvider.ANTHROPIC
            case LLMModel.MISTRAL_LARGE_LATEST:
                return LLMProvider.MISTRALAI


SUPPORTED_OPENAI_MODELS = [
    LLMModel.GPT_4_O,
    LLMModel.GPT_4_O_MINI,
    LLMModel.GPT_3_5_TURBO,
]

SUPPORTED_ANTHROPIC_MODELS = [
    LLMModel.CLAUDE_3_5_HAIKU,
    LLMModel.CLAUDE_3_5_SONNET,
    LLMModel.CLAUDE_3_OPUS_LATEST,
]

SUPPORTED_MISTRALAI_MODELS = [
    LLMModel.MISTRAL_LARGE_LATEST,
]


class AgentType(Enum):
    """Type of agent to use for executing a step."""

    TOOL_LESS = "TOOL_LESS"
    ONE_SHOT = "ONE_SHOT"
    VERIFIER = "VERIFIER"


class LogLevel(Enum):
    """Available Log Levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def is_greater_than_zero(value: int) -> int:
    """Validate greater than zero."""
    if value < 0:
        raise ValueError(f"{value} must be greater than zero")
    return value


PositiveNumber = Annotated[int, AfterValidator(is_greater_than_zero)]


class Config(BaseModel):
    """General configuration for the library."""

    # Portia Cloud Options
    portia_api_endpoint: str = "https://api.porita.dev"
    portia_api_key: SecretStr | None = None

    # LLM API Keys
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    mistralai_api_key: SecretStr | None = None

    # Storage Options
    storage_class: StorageClass
    storage_dir: str | None = None

    # Logging Options

    # default_log_level controls the minimal log level, i.e. setting to DEBUG will print all logs
    # where as setting it to ERROR will only display ERROR and above.
    default_log_level: LogLevel = LogLevel.INFO
    # default_log_sink controls where default logs are sent. By default this is STDOUT (sys.stdout)
    # but can also be set to STDERR (sys.stderr)
    # or to a file by setting this to a file path ("./logs.txt")
    default_log_sink: str = "sys.stdout"
    # json_log_serialize sets whether logs are JSON serialized before sending to the log sink.
    json_log_serialize: bool = False

    # LLM Options
    llm_provider: LLMProvider
    llm_model_name: LLMModel
    llm_model_temperature: PositiveNumber
    llm_model_seed: PositiveNumber

    # Agent Options
    default_agent_type: AgentType

    # System Context Overrides
    # Generally be mindful of context window sizes when passing additional data through these field.

    # planner_system_context_extension allows passing additional context to the
    # planner LLMs. Useful for refining instructions or passing pointers.
    planner_system_context_extension: list[str] | None = None
    # agent_system_context_extension allows passing additional context to the
    # agent LLMs. Useful for passing execution hints or other data.
    agent_system_context_extension: list[str] | None = None

    @model_validator(mode="after")
    def check_config(self) -> Config:
        """Validate Config is consistent."""
        # load keys from environment variables
        self._set_keys()
        # Portia API Key must be provided if using cloud storage
        if self.storage_class == StorageClass.CLOUD and not self.has_api_key("portia_api_key"):
            raise InvalidConfigError("portia_api_key", "Must be provided if using cloud storage")

        def validate_llm_config(expected_key: str, supported_models: list[LLMModel]) -> None:
            """Validate LLM Config."""
            if not self.has_api_key(expected_key):
                raise InvalidConfigError(
                    f"{expected_key}",
                    f"Must be provided if using {self.llm_provider}",
                )
            if self.llm_model_name not in supported_models:
                raise InvalidConfigError(
                    "llm_model_name",
                    "Unsupported model please use one of"
                    f"{", ".join(model.value for model in supported_models)}",
                )

        match self.llm_provider:
            case LLMProvider.OPENAI:
                validate_llm_config("openai_api_key", SUPPORTED_OPENAI_MODELS)
            case LLMProvider.ANTHROPIC:
                validate_llm_config("anthropic_api_key", SUPPORTED_ANTHROPIC_MODELS)
            case LLMProvider.MISTRALAI:
                validate_llm_config("mistralai_api_key", SUPPORTED_MISTRALAI_MODELS)
        return self

    def _set_keys(self) -> None:
        """Set API keys from environment variables if not already set."""
        if self.portia_api_key is None:
            self.portia_api_key = SecretStr(os.getenv("PORTIA_API_KEY") or "")
        if self.openai_api_key is None:
            self.openai_api_key = SecretStr(os.getenv("OPENAI_API_KEY") or "")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = SecretStr(os.getenv("ANTHROPIC_API_KEY") or "")
        if self.mistralai_api_key is None:
            self.mistralai_api_key = SecretStr(os.getenv("MISTRAL_API_KEY") or "")

    @classmethod
    def from_file(cls, file_path: Path) -> Config:
        """Load configuration from a JSON file."""
        with Path.open(file_path) as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def from_default(cls, **kwargs) -> Config:  # noqa: ANN003
        """Create a Config instance with default values, allowing overrides."""
        default = default_config()
        return default.model_copy(update=kwargs)

    def has_api_key(self, name: str) -> bool:
        """Check if the given API Key is available."""
        try:
            self.must_get_api_key(name)
        except InvalidConfigError:
            return False
        else:
            return True

    def must_get_api_key(self, name: str) -> SecretStr:
        """Get an api key as a SecretStr or error if not set."""
        return self.must_get(name, SecretStr)

    def must_get_raw_api_key(self, name: str) -> str:
        """Get a raw api key as a string or errors if not set."""
        key = self.must_get_api_key(name)
        return key.get_secret_value()

    def must_get(self, name: str, expected_type: type[T]) -> T:
        """Get a given value in the config ensuring a type match."""
        if not hasattr(self, name):
            raise ConfigNotFoundError(name)
        value = getattr(self, name)
        if not isinstance(value, expected_type):
            raise InvalidConfigError(name, f"Not of expected type: {expected_type}")
        # ensure non-empty values
        match value:
            case str() if value == "":
                raise InvalidConfigError(name, "Empty value not allowed")
            case SecretStr() if value.get_secret_value() == "":
                raise InvalidConfigError(name, "Empty SecretStr value not allowed")
        return value

    def swap_provider(self, provider: LLMProvider, model: LLMModel | None = None) -> None:
        """Swap the LLM provider and model. If model not provided, default model is used."""
        if provider == LLMProvider.OPENAI and self.openai_api_key is None:
            raise InvalidConfigError("openai_api_key", "Must be provided if using OpenAI")
        if provider == LLMProvider.ANTHROPIC and self.anthropic_api_key is None:
            raise InvalidConfigError("anthropic_api_key", "Must be provided if using Anthropic")
        if provider == LLMProvider.MISTRALAI and self.mistralai_api_key is None:
            raise InvalidConfigError("mistralai_api_key", "Must be provided if using MistralAI")
        self.llm_provider = provider
        if model is None:
            self.llm_model_name = self.llm_provider.default_model()
            return
        if model not in self.llm_provider.associated_models():
            raise InvalidConfigError("llm_model_name", "Unsupported model")
        self.llm_model_name = model


def default_config() -> Config:
    """Return default config."""
    return Config(
        storage_class=StorageClass.MEMORY,
        llm_provider=LLMProvider.OPENAI,
        llm_model_name=LLMModel.GPT_4_O_MINI,
        llm_model_temperature=0,
        llm_model_seed=443,
        default_agent_type=AgentType.VERIFIER,
    )
