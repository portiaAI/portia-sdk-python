"""Configuration module for the SDK.

This module defines the configuration classes and enumerations used in the SDK,
including settings for storage, API keys, LLM providers, logging, and agent options.
It also provides validation for configuration values and loading mechanisms for
config files and default settings.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Self, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from portia.errors import ConfigNotFoundError, InvalidConfigError

T = TypeVar("T")


class StorageClass(Enum):
    """Enum representing locations plans and workflows are stored.

    Attributes:
        MEMORY: Stored in memory.
        DISK: Stored on disk.
        CLOUD: Stored in the cloud.

    """

    MEMORY = "MEMORY"
    DISK = "DISK"
    CLOUD = "CLOUD"


class LLMProvider(Enum):
    """Enum for supported LLM providers.

    Attributes:
        OPENAI: OpenAI provider.
        ANTHROPIC: Anthropic provider.
        MISTRALAI: MistralAI provider.

    """

    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    MISTRALAI = "MISTRALAI"

    def associated_models(self) -> list[LLMModel]:
        """Get the associated models for the provider.

        Returns:
            list[LLMModel]: List of supported models for the provider.

        """
        match self:
            case LLMProvider.OPENAI:
                return SUPPORTED_OPENAI_MODELS
            case LLMProvider.ANTHROPIC:
                return SUPPORTED_ANTHROPIC_MODELS
            case LLMProvider.MISTRALAI:
                return SUPPORTED_MISTRALAI_MODELS

    def default_model(self) -> LLMModel:
        """Get the default model for the provider.

        Returns:
            LLMModel: The default model for the provider.

        """
        match self:
            case LLMProvider.OPENAI:
                return LLMModel.GPT_4_O_MINI
            case LLMProvider.ANTHROPIC:
                return LLMModel.CLAUDE_3_5_SONNET
            case LLMProvider.MISTRALAI:
                return LLMModel.MISTRAL_LARGE_LATEST


class LLMModel(Enum):
    """Enum for supported LLM models.

    Models are grouped by provider, with the following providers:
    - OpenAI
    - Anthropic
    - MistralAI

    Attributes:
        GPT_4_O: GPT-4 model by OpenAI.
        GPT_4_O_MINI: Mini GPT-4 model by OpenAI.
        GPT_3_5_TURBO: GPT-3.5 Turbo model by OpenAI.
        CLAUDE_3_5_SONNET: Claude 3.5 Sonnet model by Anthropic.
        CLAUDE_3_5_HAIKU: Claude 3.5 Haiku model by Anthropic.
        CLAUDE_3_OPUS_LATEST: Claude 3.0 Opus latest model by Anthropic.
        MISTRAL_LARGE_LATEST: Mistral Large Latest model by MistralAI.

    """

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
        """Get the associated provider for the model.

        Returns:
            LLMProvider: The provider associated with the model.

        """
        if self in SUPPORTED_ANTHROPIC_MODELS:
            return LLMProvider.ANTHROPIC
        if self in SUPPORTED_MISTRALAI_MODELS:
            return LLMProvider.MISTRALAI
        return LLMProvider.OPENAI


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
    """Enum for types of agents used for executing a step.

    Attributes:
        TOOL_LESS: A tool-less agent.
        ONE_SHOT: A one-shot agent.
        VERIFIER: A verifier agent.

    """

    ONE_SHOT = "ONE_SHOT"
    VERIFIER = "VERIFIER"


class PlannerType(Enum):
    """Enum for planners used for planning queries.

    Attributes:
        ONE_SHOT: A one-shot planner.

    """

    ONE_SHOT = "ONE_SHOT"
    TWO_SHOT = "TWO_SHOT"


class LogLevel(Enum):
    """Enum for available log levels.

    Attributes:
        DEBUG: Debug log level.
        INFO: Info log level.
        WARNING: Warning log level.
        ERROR: Error log level.
        CRITICAL: Critical log level.

    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def is_greater_than_zero(value: int) -> int:
    """Ensure the value is greater than zero.

    Args:
        value (int): The value to validate.

    Raises:
        ValueError: If the value is less than or equal to zero.

    Returns:
        int: The validated value.

    """
    if value < 0:
        raise ValueError(f"{value} must be greater than zero")
    return value


PositiveNumber = Annotated[int, AfterValidator(is_greater_than_zero)]


E = TypeVar("E", bound=Enum)


def parse_str_to_enum(value: str | E, enum_type: type[E]) -> E:
    """Parse a string to an enum or return the enum as is.

    Args:
        value (str | E): The value to parse.
        enum_type (type[E]): The enum type to parse the value into.

    Raises:
        InvalidConfigError: If the value cannot be parsed into the enum.

    Returns:
        E: The corresponding enum value.

    """
    if isinstance(value, str):
        try:
            return enum_type[value.upper()]
        except KeyError as e:
            raise InvalidConfigError(
                value=value,
                issue=f"Invalid value for enum {enum_type.__name__}",
            ) from e
    if isinstance(value, enum_type):
        return value

    raise InvalidConfigError(
        value=str(value),
        issue=f"Value must be a string or {enum_type.__name__}",
    )


class Config(BaseModel):
    """General configuration for the SDK.

    This class holds the configuration for the SDK, including API keys, LLM
    settings, logging options, and storage settings. It also provides validation
    for configuration consistency and offers methods for loading configuration
    from files or default values.

    Attributes:
        portia_api_endpoint: The endpoint for the Portia API.
        portia_api_key: The API key for Portia.
        openai_api_key: The API key for OpenAI.
        anthropic_api_key: The API key for Anthropic.
        mistralai_api_key: The API key for MistralAI.
        storage_class: The storage class used (e.g., MEMORY, DISK, CLOUD).
        storage_dir: The directory for storage, if applicable.
        default_log_level: The default log level (e.g., DEBUG, INFO).
        default_log_sink: The default destination for logs (e.g., sys.stdout).
        json_log_serialize: Whether to serialize logs in JSON format.
        llm_provider: The LLM provider (e.g., OpenAI, Anthropic).
        llm_model_name: The model to use for LLM tasks.
        llm_model_temperature: The temperature for LLM generation.
        llm_model_seed: The seed for LLM generation.
        default_agent_type: The default agent type.
        default_planner: The default planner type.

    """

    model_config = ConfigDict(extra="ignore")

    # Portia Cloud Options
    portia_api_endpoint: str = Field(
        default_factory=lambda: os.getenv("PORTIA_API_ENDPOINT") or "https://api.portialabs.ai",
        description="The API endpoint for the Portia Cloud API",
    )
    portia_api_key: SecretStr | None = Field(
        default_factory=lambda: SecretStr(os.getenv("PORTIA_API_KEY") or ""),
        description="The API Key for the Portia Cloud API available from the dashboard at https://app.portialabs.ai",
    )

    # LLM API Keys
    openai_api_key: SecretStr | None = Field(
        default_factory=lambda: SecretStr(os.getenv("OPENAI_API_KEY") or ""),
        description="The API Key for OpenAI. Must be set if llm-provider is OPENAI",
    )
    anthropic_api_key: SecretStr | None = Field(
        default_factory=lambda: SecretStr(os.getenv("ANTHROPIC_API_KEY") or ""),
        description="The API Key for Anthropic. Must be set if llm-provider is ANTHROPIC",
    )
    mistralai_api_key: SecretStr | None = Field(
        default_factory=lambda: SecretStr(os.getenv("MISTRAL_API_KEY") or ""),
        description="The API Key for Mistral AI. Must be set if llm-provider is MISTRALAI",
    )

    # Storage Options
    storage_class: StorageClass = Field(
        default=StorageClass.MEMORY,
        description="Where to store Plans and Workflows. By default these will be kept in memory.",
    )

    @field_validator("storage_class", mode="before")
    @classmethod
    def parse_storage_class(cls, value: str | StorageClass) -> StorageClass:
        """Parse storage class to enum if string provided."""
        return parse_str_to_enum(value, StorageClass)

    storage_dir: str | None = Field(
        default=None,
        description="If storage class is set to DISK this will be the location where plans "
        "and workflows are written in a JSON format.",
    )

    # Logging Options

    # default_log_level controls the minimal log level, i.e. setting to DEBUG will print all logs
    # where as setting it to ERROR will only display ERROR and above.
    default_log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="The log level to log at. Only respected when the default logger is used.",
    )

    @field_validator("default_log_level", mode="before")
    @classmethod
    def parse_default_log_level(cls, value: str | LogLevel) -> LogLevel:
        """Parse default_log_level to enum if string provided."""
        return parse_str_to_enum(value, LogLevel)

    # default_log_sink controls where default logs are sent. By default this is STDOUT (sys.stdout)
    # but can also be set to STDERR (sys.stderr)
    # or to a file by setting this to a file path ("./logs.txt")
    default_log_sink: str = Field(
        default="sys.stdout",
        description="Where to send logs. By default logs will be sent to sys.stdout",
    )
    # json_log_serialize sets whether logs are JSON serialized before sending to the log sink.
    json_log_serialize: bool = Field(
        default=False,
        description="Whether to serialize logs to JSON",
    )

    # LLM Options
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Which LLM Provider to use.",
    )

    @field_validator("llm_provider", mode="before")
    @classmethod
    def parse_llm_provider(cls, value: str | LLMProvider) -> LLMProvider:
        """Parse llm_provider to enum if string provided."""
        return parse_str_to_enum(value, LLMProvider)

    llm_model_name: LLMModel = Field(
        default=LLMModel.GPT_4_O_MINI,
        description="Which LLM Model to use.",
    )

    @field_validator("llm_model_name", mode="before")
    @classmethod
    def parse_llm_model_name(cls, value: str | LLMModel) -> LLMModel:
        """Parse llm_model_name to enum if string provided."""
        return parse_str_to_enum(value, LLMModel)

    llm_model_temperature: PositiveNumber = Field(
        default=0,
        description="The model temperature to use. A lower number leads to more repeatable results"
        ", a higher number more creativity.",
    )
    llm_model_seed: PositiveNumber = Field(
        default=443,
        description="The model seed to use.",
    )

    # Agent Options
    default_agent_type: AgentType = Field(
        default=AgentType.VERIFIER,
        description="The default agent type to use.",
    )

    @field_validator("default_agent_type", mode="before")
    @classmethod
    def parse_default_agent_type(cls, value: str | AgentType) -> AgentType:
        """Parse default_agent_type to enum if string provided."""
        return parse_str_to_enum(value, AgentType)

    # Planner Options
    default_planner: PlannerType = Field(
        default=PlannerType.ONE_SHOT,
        description="The default planner to use.",
    )

    @field_validator("default_planner", mode="before")
    @classmethod
    def parse_default_planner(cls, value: str | PlannerType) -> PlannerType:
        """Parse default_planner to enum if string provided."""
        return parse_str_to_enum(value, PlannerType)

    @model_validator(mode="after")
    def check_config(self) -> Self:
        """Validate Config is consistent."""
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
                    "Unsupported model please use one of: "
                    + ", ".join(model.value for model in supported_models),
                )

        match self.llm_provider:
            case LLMProvider.OPENAI:
                validate_llm_config("openai_api_key", SUPPORTED_OPENAI_MODELS)
            case LLMProvider.ANTHROPIC:
                validate_llm_config("anthropic_api_key", SUPPORTED_ANTHROPIC_MODELS)
            case LLMProvider.MISTRALAI:
                validate_llm_config("mistralai_api_key", SUPPORTED_MISTRALAI_MODELS)
        return self

    @classmethod
    def from_file(cls, file_path: Path) -> Config:
        """Load configuration from a JSON file.

        Returns:
            Config: The default config

        """
        with Path.open(file_path) as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def from_default(cls, **kwargs) -> Config:  # noqa: ANN003
        """Create a Config instance with default values, allowing overrides.

        Returns:
            Config: The default config

        """
        return default_config(**kwargs)

    def has_api_key(self, name: str) -> bool:
        """Check if the given API Key is available."""
        try:
            self.must_get_api_key(name)
        except InvalidConfigError:
            return False
        else:
            return True

    def must_get_api_key(self, name: str) -> SecretStr:
        """Retrieve the required API key for the configured provider.

        Raises:
            ConfigNotFoundError: If no API key is found for the provider.

        Returns:
            SecretStr: The required API key.

        """
        return self.must_get(name, SecretStr)

    def must_get_raw_api_key(self, name: str) -> str:
        """Retrieve the raw API key for the configured provider.

        Raises:
            ConfigNotFoundError: If no API key is found for the provider.

        Returns:
            str: The raw API key.

        """
        key = self.must_get_api_key(name)
        return key.get_secret_value()

    def must_get(self, name: str, expected_type: type[T]) -> T:
        """Retrieve any value from the config, ensuring its of the correct type.

        Args:
            name (str): The name of the config record.
            expected_type (type[T]): The expected type of the value.

        Raises:
            ConfigNotFoundError: If no API key is found for the provider.
            InvalidConfigError: If the config isn't valid

        Returns:
            T: The config value

        """
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


def default_config(**kwargs) -> Config:  # noqa: ANN003
    """Return default config with values that can be overridden.

    Returns:
        Config: The default config

    """
    return Config(
        storage_class=kwargs.pop("storage_class", StorageClass.MEMORY),
        llm_provider=kwargs.pop("llm_provider", LLMProvider.OPENAI),
        llm_model_name=kwargs.pop("llm_model_name", LLMModel.GPT_4_O_MINI),
        default_planner=kwargs.pop("default_planner", PlannerType.ONE_SHOT),
        llm_model_temperature=kwargs.pop("llm_model_temperature", 0),
        llm_model_seed=kwargs.pop("llm_model_seed", 443),
        default_agent_type=kwargs.pop("default_agent_type", AgentType.VERIFIER),
        **kwargs,
    )
