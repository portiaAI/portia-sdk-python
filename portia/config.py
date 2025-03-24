"""Configuration module for the SDK.

This module defines the configuration classes and enumerations used in the SDK,
including settings for storage, API keys, LLM providers, logging, and agent options.
It also provides validation for configuration values and loading mechanisms for
config files and default settings.
"""

from __future__ import annotations

import importlib.util
import os
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from portia.errors import ConfigNotFoundError, InvalidConfigError

if TYPE_CHECKING:
    from portia.execution_agents.base_execution_agent import BaseExecutionAgent
    from portia.planning_agents.base_planning_agent import BasePlanningAgent
else:
    # Placeholder types for runtime
    BaseExecutionAgent = Any
    BasePlanningAgent = Any


T = TypeVar("T")


class StorageClass(Enum):
    """Enum representing locations plans and runs are stored.

    Attributes:
        MEMORY: Stored in memory.
        DISK: Stored on disk.
        CLOUD: Stored in the cloud.

    """

    MEMORY = "MEMORY"
    DISK = "DISK"
    CLOUD = "CLOUD"


EXTRAS_GROUPS_DEPENDENCIES = {
    "mistral": ["mistralai", "langchain_mistralai"],
}


def validate_extras_dependencies(extra_group: str) -> None:
    """Validate that the dependencies for an extras group are installed.

    Provide a helpful error message if not all dependencies are installed.
    """

    def package_installed(package: str) -> bool:
        try:
            return importlib.util.find_spec(package) is not None
        except ImportError:
            pass
        return False

    if not all(package_installed(p) for p in EXTRAS_GROUPS_DEPENDENCIES[extra_group]):
        raise ImportError(
            f"Please install portia-sdk-python[{extra_group}] to use this functionality.",
        )


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

    def to_api_key_name(self) -> str:
        """Get the name of the API key for the provider."""
        match self:
            case LLMProvider.OPENAI:
                return "openai_api_key"
            case LLMProvider.ANTHROPIC:
                return "anthropic_api_key"
            case LLMProvider.MISTRALAI:
                return "mistralai_api_key"


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
        CLAUDE_3_OPUS: Claude 3.0 Opus model by Anthropic.
        CLAUDE_3_7_SONNET: Claude 3.7 Sonnet model by Anthropic.
        MISTRAL_LARGE: Mistral Large Latest model by MistralAI.

    """

    # OpenAI
    GPT_4_O = "gpt-4o"
    GPT_4_O_MINI = "gpt-4o-mini"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    O_3_MINI = "o3-mini"

    # Anthropic
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-latest"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-latest"
    CLAUDE_3_OPUS = "claude-3-opus-latest"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-latest"

    # MistralAI
    MISTRAL_LARGE = "mistral-large-latest"

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
    LLMModel.O_3_MINI,
]

SUPPORTED_ANTHROPIC_MODELS = [
    LLMModel.CLAUDE_3_5_HAIKU,
    LLMModel.CLAUDE_3_5_SONNET,
    LLMModel.CLAUDE_3_7_SONNET,
    LLMModel.CLAUDE_3_OPUS,
]

SUPPORTED_MISTRALAI_MODELS = [
    LLMModel.MISTRAL_LARGE,
]


class ExecutionAgentType(Enum):
    """Enum for types of agents used for executing a step.

    Attributes:
        ONE_SHOT: The one-shot agent.
        DEFAULT: The default agent.

    """

    ONE_SHOT = "ONE_SHOT"
    DEFAULT = "DEFAULT"
    CUSTOM = "CUSTOM"


class PlanningAgentType(Enum):
    """Enum for planning agents used for planning queries.

    Attributes:
        DEFAULT: The default planning agent.

    """

    DEFAULT = "DEFAULT"
    CUSTOM = "CUSTOM"


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


PLANNING_MODEL_KEY = "planning_model_name"
EXECUTION_MODEL_KEY = "execution_model_name"
INTROSPECTION_MODEL_KEY = "introspection_model_name"
LLM_TOOL_MODEL_KEY = "llm_tool_model_name"
IMAGE_TOOL_MODEL_KEY = "image_tool_model_name"
SUMMARISER_MODEL_KEY = "summariser_model_name"
DEFAULT_MODEL_KEY = "default_model_name"
PLANNING_DEFAULT_MODEL_KEY = "planning_default_model_name"

CONDITIONAL_FEATURE_FLAG = "conditional_feature_flag"


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


PLANNER_DEFAULT_MODELS = {
    LLMProvider.OPENAI: LLMModel.O_3_MINI,
    LLMProvider.ANTHROPIC: LLMModel.CLAUDE_3_5_SONNET,
    LLMProvider.MISTRALAI: LLMModel.MISTRAL_LARGE,
}

DEFAULT_MODELS = {
    LLMProvider.OPENAI: LLMModel.GPT_4_O,
    LLMProvider.ANTHROPIC: LLMModel.CLAUDE_3_5_SONNET,
    LLMProvider.MISTRALAI: LLMModel.MISTRAL_LARGE,
}


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
        llm_provider: The LLM provider.
        models: A dictionary of LLM models for each usage type.
        storage_class: The storage class used (e.g., MEMORY, DISK, CLOUD).
        storage_dir: The directory for storage, if applicable.
        default_log_level: The default log level (e.g., DEBUG, INFO).
        default_log_sink: The default destination for logs (e.g., sys.stdout).
        json_log_serialize: Whether to serialize logs in JSON format.
        planning_agent_type: The planning agent type.
        execution_agent_type: The execution agent type.

    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    # Portia Cloud Options
    portia_api_endpoint: str = Field(
        default_factory=lambda: os.getenv("PORTIA_API_ENDPOINT") or "https://api.portialabs.ai",
        description="The API endpoint for the Portia Cloud API",
    )
    portia_dashboard_url: str = Field(
        default_factory=lambda: os.getenv("PORTIA_DASHBOARD_URL") or "https://app.portialabs.ai",
        description="The URL for the Portia Cloud Dashboard",
    )
    portia_api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(os.environ["PORTIA_API_KEY"])
            if "PORTIA_API_KEY" in os.environ
            else None
        ),
        description="The API Key for the Portia Cloud API available from the dashboard at https://app.portialabs.ai",
    )

    # LLM API Keys
    openai_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("OPENAI_API_KEY") or ""),
        description="The API Key for OpenAI. Must be set if llm-provider is OPENAI",
    )
    anthropic_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("ANTHROPIC_API_KEY") or ""),
        description="The API Key for Anthropic. Must be set if llm-provider is ANTHROPIC",
    )
    mistralai_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("MISTRAL_API_KEY") or ""),
        description="The API Key for Mistral AI. Must be set if llm-provider is MISTRALAI",
    )

    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Which LLM Provider to use.",
    )

    models: dict[str, LLMModel] = Field(
        default={},
        description="A dictionary of configured LLM models for each usage.",
    )

    feature_flags: dict[str, bool] = Field(
        default={},
        description="A dictionary of feature flags for the SDK.",
    )

    @model_validator(mode="after")
    def parse_feature_flags(self) -> Self:
        """Add feature flags if not provided."""
        self.feature_flags = {
            CONDITIONAL_FEATURE_FLAG: False,
            **self.feature_flags,
        }
        return self

    @model_validator(mode="after")
    def add_default_models(self) -> Self:
        """Add default models if not provided."""
        self.models = {
            PLANNING_DEFAULT_MODEL_KEY: PLANNER_DEFAULT_MODELS[self.llm_provider],
            DEFAULT_MODEL_KEY: DEFAULT_MODELS[self.llm_provider],
            **self.models,
        }
        return self

    def model(self, usage: str) -> LLMModel:
        """Get the LLM model for the given usage."""
        if usage == PLANNING_MODEL_KEY:
            return self.models.get(PLANNING_MODEL_KEY, self.models[PLANNING_DEFAULT_MODEL_KEY])
        return self.models.get(usage, self.models[DEFAULT_MODEL_KEY])

    # Storage Options
    storage_class: StorageClass = Field(
        default_factory=lambda: StorageClass.CLOUD
        if os.getenv("PORTIA_API_KEY")
        else StorageClass.MEMORY,
        description="Where to store Plans and PlanRuns. By default these will be kept in memory"
        "if no API key is provided.",
    )

    @field_validator("storage_class", mode="before")
    @classmethod
    def parse_storage_class(cls, value: str | StorageClass) -> StorageClass:
        """Parse storage class to enum if string provided."""
        return parse_str_to_enum(value, StorageClass)

    storage_dir: str | None = Field(
        default=None,
        description="If storage class is set to DISK this will be the location where plans "
        "and runs are written in a JSON format.",
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
    # Agent Options
    execution_agent_type: ExecutionAgentType = Field(
        default=ExecutionAgentType.DEFAULT,
        description="The default agent type to use.",
    )

    @field_validator("execution_agent_type", mode="before")
    @classmethod
    def parse_execution_agent_type(cls, value: str | ExecutionAgentType) -> ExecutionAgentType:
        """Parse execution_agent_type to enum if string provided."""
        execution_agent_type = parse_str_to_enum(value, ExecutionAgentType)
        if execution_agent_type == ExecutionAgentType.CUSTOM and not cls.custom_execution_agent:
            raise InvalidConfigError(
                "execution_agent_type",
                "Custom execution agent not set",
            )
        return execution_agent_type

    # PlanningAgent Options
    planning_agent_type: PlanningAgentType = Field(
        default=PlanningAgentType.DEFAULT,
        description="The default planning_agent_type to use.",
    )

    @field_validator("planning_agent_type", mode="before")
    @classmethod
    def parse_planning_agent_type(cls, value: str | PlanningAgentType) -> PlanningAgentType:
        """Parse planning_agent_type to enum if string provided."""
        planning_agent_type = parse_str_to_enum(value, PlanningAgentType)
        if planning_agent_type == PlanningAgentType.CUSTOM and not cls.custom_planning_agent:
            raise InvalidConfigError(
                "planning_agent_type",
                "Custom planning agent not set",
            )
        return planning_agent_type

    custom_planning_agent: type[BasePlanningAgent] | None = Field(
        default=None,
        description="A custom planning agent to use.",
    )

    custom_execution_agent: type[BaseExecutionAgent] | None = Field(
        default=None,
        description="A custom execution agent to use.",
    )

    @model_validator(mode="after")
    def check_config(self) -> Self:
        """Validate Config is consistent."""
        # Portia API Key must be provided if using cloud storage
        if self.storage_class == StorageClass.CLOUD and not self.has_api_key("portia_api_key"):
            raise InvalidConfigError(
                "portia_api_key",
                "A Portia API key must be provided if using cloud storage. Follow the steps at "
                "https://docs.portialabs.ai/setup-account to obtain one if you don't already "
                "have one",
            )
        if self.storage_class == StorageClass.DISK and not self.storage_dir:
            raise InvalidConfigError(
                "storage_dir",
                "A storage directory must be provided if using disk storage",
            )

        def validate_llm_api_key(provider: LLMProvider) -> None:
            """Validate LLM Config."""
            if not self.has_api_key(provider.to_api_key_name()):
                raise InvalidConfigError(
                    f"{provider.to_api_key_name()}",
                    f"Must be provided if using {provider}",
                )

        validate_llm_api_key(self.llm_provider)
        for model in self.models.values():
            validate_llm_api_key(model.provider())

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

    def get_llm_api_key(self, model_name: LLMModel) -> SecretStr:
        """Get the API key for the given LLM model.

        Returns:
            SecretStr: The API key for the given LLM model.

        """
        match model_name.provider():
            case LLMProvider.OPENAI:
                return self.openai_api_key
            case LLMProvider.ANTHROPIC:
                return self.anthropic_api_key
            case LLMProvider.MISTRALAI:
                return self.mistralai_api_key


def llm_provider_default_from_api_keys(**kwargs) -> LLMProvider:  # noqa: ANN003
    """Get the default LLM provider from the API keys."""
    if os.getenv("OPENAI_API_KEY") or kwargs.get("openai_api_key"):
        return LLMProvider.OPENAI
    if os.getenv("ANTHROPIC_API_KEY") or kwargs.get("anthropic_api_key"):
        return LLMProvider.ANTHROPIC
    if os.getenv("MISTRAL_API_KEY") or kwargs.get("mistralai_api_key"):
        return LLMProvider.MISTRALAI
    raise InvalidConfigError(LLMProvider.OPENAI.to_api_key_name(), "No LLM API key found")


def default_config(**kwargs) -> Config:  # noqa: ANN003
    """Return default config with values that can be overridden.

    Returns:
        Config: The default config

    """
    llm_model_name = kwargs.pop("llm_model_name", None)
    models = kwargs.pop("models", {})
    for model_usage in [
        PLANNING_MODEL_KEY,
        INTROSPECTION_MODEL_KEY,
        EXECUTION_MODEL_KEY,
        LLM_TOOL_MODEL_KEY,
        IMAGE_TOOL_MODEL_KEY,
        SUMMARISER_MODEL_KEY,
    ]:
        model_name = kwargs.pop(model_usage, llm_model_name)
        if model_name and model_name not in models:
            models[model_usage] = parse_str_to_enum(model_name, LLMModel)

    llm_provider = parse_str_to_enum(
        kwargs.pop("llm_provider", llm_provider_default_from_api_keys(**kwargs)),
        LLMProvider,
    )

    default_storage_class = (
        StorageClass.CLOUD if os.getenv("PORTIA_API_KEY") else StorageClass.MEMORY
    )
    return Config(
        llm_provider=llm_provider,
        models=models,
        feature_flags=kwargs.pop("feature_flags", {}),
        storage_class=kwargs.pop("storage_class", default_storage_class),
        planning_agent_type=kwargs.pop("planning_agent_type", PlanningAgentType.DEFAULT),
        execution_agent_type=kwargs.pop("execution_agent_type", ExecutionAgentType.DEFAULT),
        **kwargs,
    )
