"""Configuration module for the SDK.

This module defines the configuration classes and enumerations used in the SDK,
including settings for storage, API keys, LLM providers, logging, and agent options.
It also provides validation for configuration values and loading mechanisms for
default settings.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Container
from enum import Enum
from typing import NamedTuple, Self, TypeVar

import tiktoken
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from portia.common import validate_extras_dependencies
from portia.errors import ConfigModelResolutionError, ConfigNotFoundError, InvalidConfigError
from portia.model import (
    AnthropicGenerativeModel,
    AzureOpenAIGenerativeModel,
    GenerativeModel,
    LangChainGenerativeModel,
    LLMProvider,
    OpenAIGenerativeModel,
)

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


class Model(NamedTuple):
    """Provider and model name tuple.

    **DEPRECATED** Use new model configuration options on Config class instead.

    Attributes:
        provider: The provider of the model.
        model_name: The name of the model in the provider's API.

    """

    provider: LLMProvider
    model_name: str


class LLMModel(Enum):
    """Enum for supported LLM models.

    **DEPRECATED** Use new model configuration options on Config class instead.

    Models are grouped by provider, with the following providers:
    - OpenAI
    - Anthropic
    - MistralAI
    - Google Generative AI
    - Azure OpenAI

    Attributes:
        GPT_4_O: GPT-4 model by OpenAI.
        GPT_4_O_MINI: Mini GPT-4 model by OpenAI.
        GPT_3_5_TURBO: GPT-3.5 Turbo model by OpenAI.
        CLAUDE_3_5_SONNET: Claude 3.5 Sonnet model by Anthropic.
        CLAUDE_3_5_HAIKU: Claude 3.5 Haiku model by Anthropic.
        CLAUDE_3_OPUS: Claude 3.0 Opus model by Anthropic.
        CLAUDE_3_7_SONNET: Claude 3.7 Sonnet model by Anthropic.
        MISTRAL_LARGE: Mistral Large Latest model by MistralAI.
        GEMINI_2_0_FLASH: Gemini 2.0 Flash model by Google Generative AI.
        GEMINI_2_0_FLASH_LITE: Gemini 2.0 Flash Lite model by Google Generative AI.
        GEMINI_1_5_FLASH: Gemini 1.5 Flash model by Google Generative AI.
        AZURE_GPT_4_O: GPT-4 model by Azure OpenAI.
        AZURE_GPT_4_O_MINI: Mini GPT-4 model by Azure OpenAI.
        AZURE_O_3_MINI: O3 Mini model by Azure OpenAI.

    Can be instantiated from a string with the following format:
        - provider/model_name  [e.g. LLMModel("openai/gpt-4o")]
        - model_name           [e.g. LLMModel("gpt-4o")]

    In the cases where the model name is not unique across providers, the earlier values in the enum
    definition will take precedence.

    """

    @classmethod
    def _missing_(cls, value: object) -> LLMModel:
        """Get the LLM model from the model name."""
        if isinstance(value, str):
            for member in cls:
                if member.api_name == value:
                    return member
                if "/" in value:
                    provider, model_name = value.split("/")
                    if (
                        member.provider().value.lower() == provider.lower()
                        and member.api_name == model_name
                    ):
                        return member
        raise ValueError(f"Invalid LLM model: {value}")

    # OpenAI
    GPT_4_O = Model(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    GPT_4_O_MINI = Model(provider=LLMProvider.OPENAI, model_name="gpt-4o-mini")
    GPT_3_5_TURBO = Model(provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo")
    O_3_MINI = Model(provider=LLMProvider.OPENAI, model_name="o3-mini")

    # Anthropic
    CLAUDE_3_5_SONNET = Model(provider=LLMProvider.ANTHROPIC, model_name="claude-3-5-sonnet-latest")
    CLAUDE_3_5_HAIKU = Model(provider=LLMProvider.ANTHROPIC, model_name="claude-3-5-haiku-latest")
    CLAUDE_3_OPUS = Model(provider=LLMProvider.ANTHROPIC, model_name="claude-3-opus-latest")
    CLAUDE_3_7_SONNET = Model(provider=LLMProvider.ANTHROPIC, model_name="claude-3-7-sonnet-latest")

    # MistralAI
    MISTRAL_LARGE = Model(provider=LLMProvider.MISTRALAI, model_name="mistral-large-latest")

    # Google Generative AI
    GEMINI_2_0_FLASH = Model(
        provider=LLMProvider.GOOGLE_GENERATIVE_AI,
        model_name="gemini-2.0-flash",
    )
    GEMINI_2_0_FLASH_LITE = Model(
        provider=LLMProvider.GOOGLE_GENERATIVE_AI,
        model_name="gemini-2.0-flash-lite",
    )
    GEMINI_1_5_FLASH = Model(
        provider=LLMProvider.GOOGLE_GENERATIVE_AI,
        model_name="gemini-1.5-flash",
    )

    # Azure OpenAI
    AZURE_GPT_4_O = Model(provider=LLMProvider.AZURE_OPENAI, model_name="gpt-4o")
    AZURE_GPT_4_O_MINI = Model(provider=LLMProvider.AZURE_OPENAI, model_name="gpt-4o-mini")
    AZURE_O_3_MINI = Model(provider=LLMProvider.AZURE_OPENAI, model_name="o3-mini")

    @property
    def api_name(self) -> str:
        """Override the default value to return the model name."""
        return self.value.model_name

    def provider(self) -> LLMProvider:
        """Get the associated provider for the model.

        Returns:
            LLMProvider: The provider associated with the model.

        """
        return self.value.provider

    def to_model_string(self) -> str:
        """Get the model string for the model.

        Returns:
            str: The model string.

        """
        return f"{self.provider().value}/{self.api_name}"


class _AllModelsSupportedWithDeprecation(Container):
    """A type that returns True for any contains check."""

    def __contains__(self, item: object) -> bool:
        """Check if the item is in the container."""
        warnings.warn(
            "Supported model checks are no longer required - any model from "
            "the provider is supported.",
            DeprecationWarning,
            stacklevel=2,
        )
        return True


ALL_MODELS_SUPPORTED_WITH_DEPRECATION = _AllModelsSupportedWithDeprecation()

SUPPORTED_OPENAI_MODELS = ALL_MODELS_SUPPORTED_WITH_DEPRECATION
SUPPORTED_ANTHROPIC_MODELS = ALL_MODELS_SUPPORTED_WITH_DEPRECATION
SUPPORTED_MISTRALAI_MODELS = ALL_MODELS_SUPPORTED_WITH_DEPRECATION
SUPPORTED_GOOGLE_GENERATIVE_AI_MODELS = ALL_MODELS_SUPPORTED_WITH_DEPRECATION
SUPPORTED_AZURE_OPENAI_MODELS = ALL_MODELS_SUPPORTED_WITH_DEPRECATION


class ExecutionAgentType(Enum):
    """Enum for types of agents used for executing a step.

    Attributes:
        ONE_SHOT: The one-shot agent.
        DEFAULT: The default agent.

    """

    ONE_SHOT = "ONE_SHOT"
    DEFAULT = "DEFAULT"


class PlanningAgentType(Enum):
    """Enum for planning agents used for planning queries.

    Attributes:
        DEFAULT: The default planning agent.

    """

    DEFAULT = "DEFAULT"


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


ALL_USAGE_KEYS = [
    PLANNING_MODEL_KEY := "planning_model",
    EXECUTION_MODEL_KEY := "execution_model",
    INTROSPECTION_MODEL_KEY := "introspection_model",
    SUMMARISER_MODEL_KEY := "summariser_model",
    DEFAULT_MODEL_KEY := "default_model",
]

FEATURE_FLAG_AGENT_MEMORY_ENABLED = "feature_flag_agent_memory_enabled"


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


PROVIDER_DEFAULT_MODELS = {
    PLANNING_MODEL_KEY: {
        LLMProvider.OPENAI: "openai/o3-mini",
        LLMProvider.ANTHROPIC: "anthropic/claude-3-7-sonnet-latest",
        LLMProvider.MISTRALAI: "mistralai/mistral-large-latest",
        LLMProvider.GOOGLE_GENERATIVE_AI: "google/gemini-2.0-flash",
        LLMProvider.AZURE_OPENAI: "azure-openai/o3-mini",
    },
    DEFAULT_MODEL_KEY: {
        LLMProvider.OPENAI: "openai/gpt-4o",
        LLMProvider.ANTHROPIC: "anthropic/claude-3-7-sonnet-latest",
        LLMProvider.MISTRALAI: "mistralai/mistral-large-latest",
        LLMProvider.GOOGLE_GENERATIVE_AI: "google/gemini-2.0-flash",
        LLMProvider.AZURE_OPENAI: "azure-openai/gpt-4o",
    },
}

PLANNER_DEFAULT_MODELS = PROVIDER_DEFAULT_MODELS[PLANNING_MODEL_KEY]

DEFAULT_MODELS = PROVIDER_DEFAULT_MODELS[DEFAULT_MODEL_KEY]


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
        google_api_key: The API key for Google Generative AI.
        azure_openai_api_key: The API key for Azure OpenAI.
        azure_openai_endpoint: The endpoint for Azure OpenAI.
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

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

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
            SecretStr(os.environ["PORTIA_API_KEY"]) if "PORTIA_API_KEY" in os.environ else None
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
    google_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("GOOGLE_API_KEY") or ""),
        description="The API Key for Google Generative AI. Must be set if llm-provider is "
        "GOOGLE_GENERATIVE_AI",
    )
    azure_openai_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("AZURE_OPENAI_API_KEY") or ""),
        description="The API Key for Azure OpenAI. Must be set if llm-provider is AZURE_OPENAI",
    )
    azure_openai_endpoint: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT") or "",
        description="The endpoint for Azure OpenAI. Must be set if llm-provider is AZURE_OPENAI",
    )

    llm_provider: LLMProvider | None = Field(
        default=None,
        description="Which LLM Provider to use for resolving GenerativeModels. Can be None if "
        "GenerativeModel instances are provided directly.",
    )

    models: dict[str, str | GenerativeModel] = Field(
        default_factory=dict,
        description="A dictionary of configured LLM models for each usage.",
    )

    @field_validator("models", mode="before")
    @classmethod
    def parse_models(
        cls,
        value: dict[str, LLMModel | str | GenerativeModel],
    ) -> dict[str, GenerativeModel | str]:
        """Convert legacy LLMModel values to str with deprecation warning."""
        new_models = {}
        for key, model_value in value.items():
            new_model_value = model_value
            if isinstance(model_value, LLMModel):
                warnings.warn(
                    "LLMModel values are deprecated and will be removed in a future version.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                new_model_value = model_value.to_model_string()
            new_models[key] = new_model_value
        return new_models

    feature_flags: dict[str, bool] = Field(
        default={},
        description="A dictionary of feature flags for the SDK.",
    )

    @model_validator(mode="after")
    def parse_feature_flags(self) -> Self:
        """Add feature flags if not provided."""
        self.feature_flags = {
            # Fill here with any default feature flags.
            # e.g. CONDITIONAL_FLAG: True,
            FEATURE_FLAG_AGENT_MEMORY_ENABLED: False,
            **self.feature_flags,
        }
        return self

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
        return parse_str_to_enum(value, ExecutionAgentType)

    # PlanningAgent Options
    planning_agent_type: PlanningAgentType = Field(
        default=PlanningAgentType.DEFAULT,
        description="The default planning_agent_type to use.",
    )

    @field_validator("planning_agent_type", mode="before")
    @classmethod
    def parse_planning_agent_type(cls, value: str | PlanningAgentType) -> PlanningAgentType:
        """Parse planning_agent_type to enum if string provided."""
        return parse_str_to_enum(value, PlanningAgentType)

    large_output_threshold_tokens: int = Field(
        default=1_000,
        description="The threshold number of tokens before we start treating an output as a"
        "large output and write it to agent memory rather than storing it locally",
    )

    def exceeds_output_threshold(self, value: str | list[str | dict]) -> bool:
        """Determine whether the provided output value exceeds the large output threshold."""
        if not self.feature_flags.get(FEATURE_FLAG_AGENT_MEMORY_ENABLED):
            return False
        # It doesn't really matter which model we use here, so choose gpt2 for speed.
        # More details at https://chatgpt.com/share/67ee4931-a794-8007-9859-13aca611dba9
        encoding = tiktoken.get_encoding("gpt2").encode(str(value))
        return len(encoding) > self.large_output_threshold_tokens

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

        # Model config validation. Either llm_provider or default_model must be set.
        if self.llm_provider is None and DEFAULT_MODEL_KEY not in self.models:
            raise InvalidConfigError(
                "llm_provider or default_model",
                "Either llm_provider or default_model must be set",
            )
        # Check default_model can be resolved.
        _ = self.resolve_model()
        # Check that all models passed as strings are instantiable, i.e. they have the
        # right API keys and other required configuration.
        for model in self.models.values():
            if isinstance(model, str):
                self._parse_model_string(model)

        return self

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

    def model(self, usage: str) -> GenerativeModel:
        """Get a model from the config.

        **DEPRECATED** Use Config.resolve_model instead.
        """
        warnings.warn(
            "The model method is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.resolve_model(usage)

    def resolve_model(self, usage: str = DEFAULT_MODEL_KEY) -> GenerativeModel:
        """Resolve the default model from the config.

        The order of precedence is:
        1. A directly provided model object or string for the usage
        2. The default model for the LLM provider for the usage
        3. The default model

        If none of these can be satisfied, a ConfigModelResolutionError is raised.

        Args:
            usage (str): The usage of the model. Defaults to the default model key.

        Returns:
            GenerativeModel: The default model object.

        Raises:
            InvalidConfigError: If no default model is provided or can be resolved.

        """
        if usage not in [*ALL_USAGE_KEYS, *self.models.keys()]:
            raise ConfigModelResolutionError(
                f"Invalid usage: {usage!r}. Must be one of {ALL_USAGE_KEYS} or a key in the "
                "models dictionary.",
            )
        model: str | GenerativeModel | None = self.models.get(usage)
        if model and isinstance(model, GenerativeModel):
            return model
        if model and isinstance(model, str):
            return self._parse_model_string(model)
        if (
            self.llm_provider
            and usage in PROVIDER_DEFAULT_MODELS
            and self.llm_provider in PROVIDER_DEFAULT_MODELS[usage]
        ):
            model = PROVIDER_DEFAULT_MODELS[usage][self.llm_provider]
            return self._parse_model_string(model)
        if usage != DEFAULT_MODEL_KEY:
            try:
                # Try to return the default model
                return self.resolve_model(DEFAULT_MODEL_KEY)
            except ConfigModelResolutionError:
                # This should not happen due to Config model validation
                # at instantiation time.
                pass

        raise ConfigModelResolutionError(
            f"Model could not be resolved for usage {usage!r}. Either an LLM Provider must be set, "
            "a model must be provided for the usage, or a default model must be set.",
        )

    def resolve_langchain_model(self, usage: str = DEFAULT_MODEL_KEY) -> LangChainGenerativeModel:
        """Resolve a LangChain model from the default model configuration.

        Returns:
            LangChainGenerativeModel: The LangChain GenerativeModel object.

        Raises:
            TypeError: If the resolved model is not a LangChainGenerativeModel.

        """
        model = self.resolve_model(usage)
        if isinstance(model, LangChainGenerativeModel):
            return model
        raise TypeError(
            f"A LangChainGenerativeModel is required, but the config for "
            f"{usage} resolved to {model}.",
        )

    def _parse_model_string(self, model_string: str) -> GenerativeModel:
        """Parse a model string in the form of "provider-prefix/model_name` to a GenerativeModel.

        Supported provider-prefixes are:
        - openai
        - anthropic
        - mistral (requires portia-sdk-python[mistral] to be installed)
        - google (requires portia-sdk-python[google] to be installed)
        - azure-openai

        Args:
            model_string (str): The model string to parse. E.G. "openai/gpt-4o"

        Returns:
            GenerativeModel: The parsed model.

        """
        provider, model_name = model_string.strip().split("/", maxsplit=1)
        llm_provider = LLMProvider(provider)
        return self._construct_model(llm_provider, model_name)

    def _construct_model(self, llm_provider: LLMProvider, model_name: str) -> GenerativeModel:
        """Construct a Model instance from an LLMProvider and model name."""
        match llm_provider:
            case LLMProvider.OPENAI:
                return OpenAIGenerativeModel(
                    model_name=model_name,
                    api_key=self.must_get_api_key("openai_api_key"),
                )
            case LLMProvider.ANTHROPIC:
                return AnthropicGenerativeModel(
                    model_name=model_name,
                    api_key=self.must_get_api_key("anthropic_api_key"),
                )
            case LLMProvider.MISTRALAI:
                validate_extras_dependencies("mistralai")
                from portia.model import MistralAIGenerativeModel

                return MistralAIGenerativeModel(
                    model_name=model_name,
                    api_key=self.must_get_api_key("mistralai_api_key"),
                )
            case LLMProvider.GOOGLE_GENERATIVE_AI:
                validate_extras_dependencies("google")
                from portia.model import GoogleGenAiGenerativeModel

                return GoogleGenAiGenerativeModel(
                    model_name=model_name,
                    api_key=self.must_get_api_key("google_api_key"),
                )
            case LLMProvider.AZURE_OPENAI:
                return AzureOpenAIGenerativeModel(
                    model_name=model_name,
                    api_key=self.azure_openai_api_key,
                    azure_endpoint=self.azure_openai_endpoint,
                )
            case LLMProvider.CUSTOM:
                raise ValueError(f"Cannot construct a custom model from a string {model_name}")


def llm_provider_default_from_api_keys(**kwargs) -> LLMProvider | None:  # noqa: ANN003
    """Get the default LLM provider from the API keys.

    Returns:
        LLMProvider: The default LLM provider.
        None: If no API key is found.

    """
    if os.getenv("OPENAI_API_KEY") or kwargs.get("openai_api_key"):
        return LLMProvider.OPENAI
    if os.getenv("ANTHROPIC_API_KEY") or kwargs.get("anthropic_api_key"):
        return LLMProvider.ANTHROPIC
    if os.getenv("MISTRAL_API_KEY") or kwargs.get("mistralai_api_key"):
        return LLMProvider.MISTRALAI
    if os.getenv("GOOGLE_API_KEY") or kwargs.get("google_api_key"):
        return LLMProvider.GOOGLE_GENERATIVE_AI
    if (os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")) or (
        kwargs.get("azure_openai_api_key") and kwargs.get("azure_openai_endpoint")
    ):
        return LLMProvider.AZURE_OPENAI
    return None


def default_config(**kwargs) -> Config:  # noqa: ANN003
    """Return default config with values that can be overridden.

    Returns:
        Config: The default config

    """
    models = kwargs.pop("models", {})
    # Handle models passed directly as keyword arguments rather than in the models dictionary
    for model_usage in ALL_USAGE_KEYS:
        model_name = kwargs.pop(model_usage, None)
        if model_name and model_usage not in models:
            models[model_usage] = model_name
        elif model_name and model_usage in models and models[model_usage] != model_name:
            raise InvalidConfigError(
                value=model_usage,
                issue=f"Model for usage {model_usage} is specified both as a keyword argument and "
                "in the models dictionary.",
            )
    # Handle deprecated llm_model_name keyword argument
    if llm_model_name := kwargs.pop("llm_model_name", None):
        warnings.warn(
            "llm_model_name is deprecated and will be removed in a future version. Use "
            "'default_model' instead.",
            stacklevel=2,
            category=DeprecationWarning,
        )
        if DEFAULT_MODEL_KEY not in models:
            models[DEFAULT_MODEL_KEY] = llm_model_name

    inferred_llm_provider = llm_provider_default_from_api_keys(**kwargs)
    if "llm_provider" in kwargs or inferred_llm_provider:
        llm_provider = parse_str_to_enum(
            kwargs.pop("llm_provider", inferred_llm_provider),
            LLMProvider,
        )
    else:
        llm_provider = None

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
