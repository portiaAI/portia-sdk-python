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
from typing import Any, NamedTuple, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from portia.common import validate_extras_dependencies
from portia.errors import ConfigNotFoundError, InvalidConfigError
from portia.logger import logger
from portia.model import (
    AnthropicGenerativeModel,
    AzureOpenAIGenerativeModel,
    GenerativeModel,
    LangChainGenerativeModel,
    LLMProvider,
    OpenAIGenerativeModel,
    OpenRouterGenerativeModel,
)
from portia.token_check import estimate_tokens

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
    GPT_4_1 = Model(provider=LLMProvider.OPENAI, model_name="gpt-4.1")
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
    GEMINI_2_5_FLASH = Model(
        provider=LLMProvider.GOOGLE,
        model_name="gemini-2.5-flash-preview-04-17",
    )
    GEMINI_2_5_PRO = Model(
        provider=LLMProvider.GOOGLE,
        model_name="gemini-2.5-pro",
    )
    GEMINI_2_0_FLASH = Model(
        provider=LLMProvider.GOOGLE,
        model_name="gemini-2.0-flash",
    )
    GEMINI_2_0_FLASH_LITE = Model(
        provider=LLMProvider.GOOGLE,
        model_name="gemini-2.0-flash-lite",
    )
    GEMINI_1_5_FLASH = Model(
        provider=LLMProvider.GOOGLE,
        model_name="gemini-1.5-flash",
    )

    # Azure OpenAI
    AZURE_GPT_4_O = Model(provider=LLMProvider.AZURE_OPENAI, model_name="gpt-4o")
    AZURE_GPT_4_O_MINI = Model(provider=LLMProvider.AZURE_OPENAI, model_name="gpt-4o-mini")
    AZURE_GPT_4_1 = Model(provider=LLMProvider.AZURE_OPENAI, model_name="gpt-4.1")
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
        TRACE: Trace log level (very verbose; below DEBUG).
        DEBUG: Debug log level.
        INFO: Info log level.
        WARNING: Warning log level.
        ERROR: Error log level.
        CRITICAL: Critical log level.

    """

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


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
            return enum_type[value.upper().replace("-", "_")]
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


PLANNING_MODEL_KEY = "planning_model_name"
EXECUTION_MODEL_KEY = "execution_model_name"
INTROSPECTION_MODEL_KEY = "introspection_model_name"
SUMMARISER_MODEL_KEY = "summariser_model_name"
DEFAULT_MODEL_KEY = "default_model_name"

MODEL_EXTRA_KWARGS = {
    "openai/o3-mini": {"reasoning_effort": "medium"},
    "openai/o4-mini": {"reasoning_effort": "medium"},
    "anthropic/claude-3-7-sonnet-latest": {
        "model_kwargs": {"thinking": {"type": "enabled", "budget_tokens": 3000}}
    },
}


class GenerativeModelsConfig(BaseModel):
    """Configuration for a Generative Models.

    These models do not all need to be specified manually. If an LLM provider is configured,
    Portia will use default models that are selected for the particular use-case.

    Attributes:
        default_model: The default generative model to use. This model is used as the fallback
            model if no other model is specified. It is also used by default in the Portia SDK
            tool that require an LLM.

        planning_model: The model to use for the PlanningAgent. Reasoning models are a good choice
            here, as they are able to reason about the problem and the possible solutions. If not
            specified, the default_model will be used.

        execution_model: The model to use for the ExecutionAgent. This model is used for the
            distilling context from the plan run into tool calls. If not specified, the
            default_model will be used.

        introspection_model: The model to use for the IntrospectionAgent. This model is used to
            introspect the problem and the plan. If not specified, the default_model will be used.

        summarizer_model: The model to use for the SummarizerAgent. This model is used to
            summarize output from the plan run. If not specified, the default_model will be used.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_model: GenerativeModel | str | None = None
    planning_model: GenerativeModel | str | None = None
    execution_model: GenerativeModel | str | None = None
    introspection_model: GenerativeModel | str | None = None
    summarizer_model: GenerativeModel | str | None = None

    @model_validator(mode="before")
    @classmethod
    def parse_models(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert legacy LLMModel values to str with deprecation warning."""
        new_data = {}
        for key, value in data.items():
            if isinstance(value, LLMModel):
                warnings.warn(
                    "LLMModel values are deprecated and will be removed in a future version.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                new_data[key] = value.to_model_string()
            else:
                new_data[key] = value
        return new_data


# By default, cache for 1 day. Most of our prompts have the current date in them, so we don't
# need to cache for longer than a day.
CACHE_TTL_SECONDS = 60 * 60 * 24


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
        aws_access_key_id: The AWS access key ID.
        aws_secret_access_key: The AWS secret access key.
        aws_default_region: The AWS default region.
        aws_credentials_profile_name: The AWS credentials profile name.
        azure_openai_api_key: The API key for Azure OpenAI.
        azure_openai_endpoint: The endpoint for Azure OpenAI.
        llm_provider: The LLM provider. If set, Portia uses this to select the best models
            for each agent. Can be None if custom models are provided.
        models: A configuration for the LLM models for Portia to use.
        storage_class: The storage class used (e.g., MEMORY, DISK, CLOUD).
        storage_dir: The directory for storage, if applicable.
        default_log_level: The default log level (e.g., DEBUG, INFO).
        default_log_sink: The default destination for logs (e.g., sys.stdout).
        json_log_serialize: Whether to serialize logs in JSON format.
        planning_agent_type: The planning agent type.
        execution_agent_type: The execution agent type.
        feature_flags: A dictionary of feature flags for the SDK.
        clarifications_enabled: Whether to enable clarifications for the execution agent.

    """

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
    openrouter_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("OPENROUTER_API_KEY") or ""),
        description="The API Key for OpenRouter. Must be set if llm-provider is OPENROUTER",
    )
    groq_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("GROQ_API_KEY") or ""),
        description="The API Key for Groq. Must be set if llm-provider is GROQ",
    )
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
        description="The API Key for Google Generative AI. Must be set if llm-provider is GOOGLE",
    )
    azure_openai_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("AZURE_OPENAI_API_KEY") or ""),
        description="The API Key for Azure OpenAI. Must be set if llm-provider is AZURE_OPENAI",
    )
    azure_openai_endpoint: str = Field(
        default_factory=lambda: (os.getenv("AZURE_OPENAI_ENDPOINT") or ""),
        description="The endpoint for Azure OpenAI. Must be set if llm-provider is AZURE_OPENAI",
    )
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1",
        description="The base URL for Ollama. Must be set if llm-provider is OLLAMA",
    )
    aws_access_key_id: str = Field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID") or "",
        description="The AWS access key ID. Must be set if llm-provider is AMAZON",
    )
    aws_secret_access_key: str = Field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY") or "",
        description="The AWS secret access key. Must be set if llm-provider is AMAZON",
    )
    aws_default_region: str = Field(
        default_factory=lambda: os.getenv("AWS_DEFAULT_REGION") or "",
        description="The AWS default region. Must be set if llm-provider is AMAZON",
    )
    aws_credentials_profile_name: str | None = Field(
        default_factory=lambda: os.getenv("AWS_CREDENTIALS_PROFILE_NAME") or None,
        description=(
            "The AWS credentials profile name. Must be set if llm-provider is AMAZON, "
            "if not provided, aws_access_key_id and aws_secret_access_key must be provided"
        ),
    )

    llm_redis_cache_url: str | None = Field(
        default_factory=lambda: os.getenv("LLM_REDIS_CACHE_URL"),
        description="Optional Redis URL used for caching LLM responses. This URl should include "
        "the auth details if required for access to the cache.",
    )

    llm_provider: LLMProvider | None = Field(
        default=None,
        description="The LLM (API) provider. If set, Portia uses this to select the "
        " best models for each agent. Can be None if custom models are provided.",
    )

    models: GenerativeModelsConfig = Field(
        default_factory=lambda: GenerativeModelsConfig(),
        description="Manual configuration for the generative models for Portia to use for "
        "different agents. See the GenerativeModels class for more information.",
    )

    feature_flags: dict[str, bool] = Field(
        default={},
        description="A dictionary of feature flags for the SDK.",
    )
    argument_clarifications_enabled: bool = Field(
        default=False,
        description=(
            "Whether to enable clarifications for the execution agent which allows the agent to "
            "ask clarifying questions to the user about the arguments to a tool call."
        ),
    )

    @model_validator(mode="after")
    def parse_feature_flags(self) -> Self:
        """Add feature flags if not provided."""
        self.feature_flags = {
            # Fill here with any default feature flags.
            # e.g. CONDITIONAL_FLAG: True,
            FEATURE_FLAG_AGENT_MEMORY_ENABLED: True,
            **self.feature_flags,
        }
        return self

    @model_validator(mode="after")
    def setup_cache(self) -> Self:
        """Set up LLM cache if Redis URL is provided."""
        if self.llm_redis_cache_url and validate_extras_dependencies("cache", raise_error=False):
            from langchain_redis import RedisCache

            cache = RedisCache(self.llm_redis_cache_url, ttl=CACHE_TTL_SECONDS, prefix="llm:")
            LangChainGenerativeModel.set_cache(cache)
        elif self.llm_redis_cache_url:
            logger().warning(  # pragma: no cover
                "Not using cache as cache group is not installed. "  # pragma: no cover
                "Install portia-sdk-python[caching] to use caching."  # pragma: no cover
            )  # pragma: no cover
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
        default=ExecutionAgentType.ONE_SHOT,
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
        return estimate_tokens(str(value)) > self.large_output_threshold_tokens

    def get_agent_default_model(  # noqa: C901, PLR0911, PLR0912
        self,
        agent_key: str,
        llm_provider: LLMProvider | None = None,
    ) -> GenerativeModel | str | None:
        """Get the default model for the given agent key."""
        match agent_key:
            case "planning_model":
                match llm_provider:
                    case LLMProvider.OPENAI:
                        return "openai/o3-mini"
                    case LLMProvider.ANTHROPIC:
                        return "anthropic/claude-3-7-sonnet-latest"
                    case LLMProvider.MISTRALAI:
                        return "mistralai/mistral-large-latest"
                    case LLMProvider.GOOGLE:
                        return "google/gemini-2.5-pro"
                    case LLMProvider.AMAZON:
                        return "amazon/eu.anthropic.claude-3-7-sonnet-20250219-v1:0"
                    case LLMProvider.AZURE_OPENAI:
                        return "azure-openai/o3-mini"
                    case LLMProvider.OPENROUTER:
                        return "openrouter/moonshotai/kimi-k2"
                    case LLMProvider.GROQ:
                        return "groq/llama3-70b-8192"
                return None
            case "introspection_model":
                match llm_provider:
                    case LLMProvider.OPENAI:
                        return "openai/o4-mini"
                    case LLMProvider.ANTHROPIC:
                        return "anthropic/claude-3-7-sonnet-latest"
                    case LLMProvider.MISTRALAI:
                        return "mistralai/mistral-large-latest"
                    case LLMProvider.GOOGLE:
                        return "google/gemini-2.5-flash"
                    case LLMProvider.AMAZON:
                        return "amazon/eu.anthropic.claude-3-7-sonnet-20250219-v1:0"
                    case LLMProvider.AZURE_OPENAI:
                        return "azure-openai/o4-mini"
                    case LLMProvider.GROQ:
                        return "groq/llama3-8b-8192"
                return None
            case "default_model":
                match llm_provider:
                    case LLMProvider.OPENAI:
                        return "openai/gpt-4.1"
                    case LLMProvider.ANTHROPIC:
                        return "anthropic/claude-3-5-sonnet-latest"
                    case LLMProvider.MISTRALAI:
                        return "mistralai/mistral-large-latest"
                    case LLMProvider.GOOGLE:
                        return "google/gemini-2.5-flash"
                    case LLMProvider.AMAZON:
                        return "amazon/eu.anthropic.claude-3-7-sonnet-20250219-v1:0"
                    case LLMProvider.AZURE_OPENAI:
                        return "azure-openai/gpt-4.1"
                    case LLMProvider.OPENROUTER:
                        return "openrouter/moonshotai/kimi-k2"
                    case LLMProvider.GROQ:
                        return "groq/llama3-8b-8192"
                return None

    @model_validator(mode="after")
    def fill_default_models(self) -> Self:
        """Fill in default models for the LLM provider if not provided."""
        if self.models.default_model is None and (
            model := self.get_agent_default_model(
                agent_key="default_model",
                llm_provider=self.llm_provider,
            )
        ):
            self.models.default_model = model
        if self.models.default_model is None:
            raise InvalidConfigError(
                "llm_provider or default_model",
                "Either llm_provider must be set, default model must be set, or an API key must be "
                "provided to allow for automatic model selection. If you are expecting to use an "
                "external LLM provider (e.g. OpenAI / Anthropic etc), make sure you have provided "
                "an API key for that provider.",
            )
        if self.models.planning_model is None and (
            model := self.get_agent_default_model(
                agent_key="planning_model",
                llm_provider=self.llm_provider,
            )
        ):
            self.models.planning_model = model
        if self.models.introspection_model is None and (
            model := self.get_agent_default_model(
                agent_key="introspection_model",
                llm_provider=self.llm_provider,
            )
        ):
            self.models.introspection_model = model
        return self

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

        # Check that all models passed as strings are instantiable, i.e. they have the
        # right API keys and other required configuration.
        for model_getter, label, value in [
            (self.get_default_model, "default_model", self.models.default_model),
            (self.get_planning_model, "planning_model", self.models.planning_model),
            (self.get_execution_model, "execution_model", self.models.execution_model),
            (self.get_introspection_model, "introspection_model", self.models.introspection_model),
            (
                self.get_summarizer_model,
                "summarizer_model",
                self.models.summarizer_model,
            ),
        ]:
            try:
                model_getter()
            except (ImportError, ConfigNotFoundError, ValueError):
                raise
            except Exception as e:
                raise InvalidConfigError(
                    label,
                    f"The value {value!s} is not valid for the the {label} model. "
                    "This is usually either because of a typo in the model name or a missing "
                    "API Key. Please review the docs here: https://docs.portialabs.ai/manage-config",
                ) from e
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

    def get_default_model(self) -> GenerativeModel:
        """Get or build the default model from the config.

        The default model will always be present. It is a general purpose model that is used
        for the SDK's LLM-based Tools, such as the ImageUnderstandingTool and the LLMTool.

        Additionally, unless specified all other specific agent models will default to this model.
        """
        model = self.get_generative_model(self.models.default_model)
        if model is None:
            # Default model is required, but not provided.
            raise InvalidConfigError(
                "default_model",
                "A default model must be set",
            )
        return model

    def get_planning_model(self) -> GenerativeModel:
        """Get or build the planning model from the config.

        See the GenerativeModelsConfig class for more information
        """
        return self.get_generative_model(self.models.planning_model) or self.get_default_model()

    def get_execution_model(self) -> GenerativeModel:
        """Get or build the execution model from the config.

        See the GenerativeModelsConfig class for more information
        """
        return self.get_generative_model(self.models.execution_model) or self.get_default_model()

    def get_introspection_model(self) -> GenerativeModel:
        """Get or build the introspection model from the config.

        See the GenerativeModelsConfig class for more information
        """
        return (
            self.get_generative_model(self.models.introspection_model) or self.get_default_model()
        )

    def get_summarizer_model(self) -> GenerativeModel:
        """Get or build the summarizer model from the config.

        See the GenerativeModelsConfig class for more information
        """
        return self.get_generative_model(self.models.summarizer_model) or self.get_default_model()

    def get_generative_model(
        self,
        model: str | GenerativeModel | None,
    ) -> GenerativeModel | None:
        """Get a GenerativeModel instance.

        Args:
            model (str | GenerativeModel | None): The model to get, either specified as a
                string in the form of "provider/model_name", or as a GenerativeModel instance.
                Also accepts None, in which case None is returned.

        Returns:
            GenerativeModel | None: The model instance or None.

        """
        if model is None:
            return None
        if isinstance(model, str):
            return self._parse_model_string(model)
        return model

    def _parse_model_string(self, model_string: str) -> GenerativeModel:
        """Parse a model string in the form of "provider-prefix/model_name" to a GenerativeModel.

        Supported provider-prefixes are:
        - openai
        - anthropic
        - mistralai (requires portia-sdk-python[mistral] to be installed)
        - google (requires portia-sdk-python[google] to be installed)
        - azure-openai

        Args:
            model_string (str): The model string to parse. E.G. "openai/gpt-4o"

        Returns:
            GenerativeModel: The parsed model.

        """
        parts = model_string.strip().split("/", maxsplit=1)
        provider = parts[0]
        model_name = parts[1]

        llm_provider = LLMProvider(provider)
        return self._construct_model_from_name(llm_provider, model_name)

    def _construct_model_from_name(  # noqa: PLR0911,C901
        self,
        llm_provider: LLMProvider,
        model_name: str,
    ) -> GenerativeModel:
        """Construct a Model instance from an LLMProvider and model name.

        Args:
            llm_provider (LLMProvider): The LLM provider.
            model_name (str): The model name as it appears in the LLM provider's API.

        Returns:
            GenerativeModel: The constructed model.

        """
        match llm_provider:
            case LLMProvider.OPENROUTER:
                return OpenRouterGenerativeModel(
                    model_name=model_name,
                    api_key=self.must_get_api_key("openrouter_api_key"),
                    **MODEL_EXTRA_KWARGS.get(f"{llm_provider.value}/{model_name}", {}),
                )
            case LLMProvider.GROQ:
                validate_extras_dependencies("groq", raise_error=True)
                from portia.model import GroqGenerativeModel

                return GroqGenerativeModel(
                    model_name=model_name,
                    api_key=self.must_get_api_key("groq_api_key"),
                    **MODEL_EXTRA_KWARGS.get(f"{llm_provider.value}/{model_name}", {}),
                )
            case LLMProvider.OPENAI:
                return OpenAIGenerativeModel(
                    model_name=model_name,
                    api_key=self.must_get_api_key("openai_api_key"),
                    **MODEL_EXTRA_KWARGS.get(f"{llm_provider.value}/{model_name}", {}),
                )
            case LLMProvider.ANTHROPIC:
                return AnthropicGenerativeModel(
                    model_name=model_name,
                    api_key=self.must_get_api_key("anthropic_api_key"),
                    **MODEL_EXTRA_KWARGS.get(f"{llm_provider.value}/{model_name}", {}),
                )
            case LLMProvider.MISTRALAI:
                validate_extras_dependencies("mistralai", raise_error=True)
                from portia.model import MistralAIGenerativeModel

                return MistralAIGenerativeModel(
                    model_name=model_name,
                    api_key=self.must_get_api_key("mistralai_api_key"),
                    **MODEL_EXTRA_KWARGS.get(f"{llm_provider.value}/{model_name}", {}),
                )
            case LLMProvider.GOOGLE | LLMProvider.GOOGLE_GENERATIVE_AI:
                validate_extras_dependencies("google", raise_error=True)
                from portia.model import GoogleGenAiGenerativeModel

                return GoogleGenAiGenerativeModel(
                    model_name=model_name,
                    api_key=self.must_get_api_key("google_api_key"),
                    **MODEL_EXTRA_KWARGS.get(f"{llm_provider.value}/{model_name}", {}),
                )
            case LLMProvider.AMAZON:
                validate_extras_dependencies("amazon", raise_error=True)
                from portia.model import AmazonBedrockGenerativeModel

                return AmazonBedrockGenerativeModel(
                    model_id=model_name,
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.aws_default_region,
                    credentials_profile_name=self.aws_credentials_profile_name,
                    **MODEL_EXTRA_KWARGS.get(f"{llm_provider.value}/{model_name}", {}),
                )
            case LLMProvider.AZURE_OPENAI:
                return AzureOpenAIGenerativeModel(
                    model_name=model_name,
                    api_key=self.must_get_api_key("azure_openai_api_key"),
                    azure_endpoint=self.must_get("azure_openai_endpoint", str),
                    **MODEL_EXTRA_KWARGS.get(f"{llm_provider.value}/{model_name}", {}),
                )
            case LLMProvider.OLLAMA:
                validate_extras_dependencies("ollama", raise_error=True)
                from portia.model import OllamaGenerativeModel

                return OllamaGenerativeModel(
                    model_name=model_name,
                    base_url=self.ollama_base_url,
                    **MODEL_EXTRA_KWARGS.get(f"{llm_provider.value}/{model_name}", {}),
                )
            case LLMProvider.CUSTOM:
                raise ValueError(f"Cannot construct a custom model from a string {model_name}")


def llm_provider_default_from_api_keys(**kwargs) -> LLMProvider | None:  # noqa: ANN003, PLR0911
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
        return LLMProvider.GOOGLE
    if (
        os.getenv("AWS_ACCESS_KEY_ID")
        or kwargs.get("aws_access_key_id")
        or kwargs.get("aws_credentials_profile_name")
        or os.getenv("AWS_CREDENTIALS_PROFILE_NAME")
    ):
        return LLMProvider.AMAZON
    if (os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")) or (
        kwargs.get("azure_openai_api_key") and kwargs.get("azure_openai_endpoint")
    ):
        return LLMProvider.AZURE_OPENAI
    if os.getenv("OPENROUTER_API_KEY") or kwargs.get("openrouter_api_key"):
        return LLMProvider.OPENROUTER
    if os.getenv("GROQ_API_KEY") or kwargs.get("groq_api_key"):
        return LLMProvider.GROQ
    return None


def default_config(**kwargs) -> Config:  # noqa: ANN003
    """Return default config with values that can be overridden.

    Returns:
        Config: The default config

    """
    llm_provider_from_api_keys = llm_provider_default_from_api_keys(**kwargs)
    if "llm_provider" in kwargs and (kwargs_provider := kwargs.pop("llm_provider")) is not None:
        llm_provider = parse_str_to_enum(
            kwargs_provider,
            LLMProvider,
        )
    elif llm_provider_from_api_keys:
        llm_provider = llm_provider_from_api_keys
    else:
        warnings.warn("No API keys found for any LLM provider", stacklevel=2, category=UserWarning)
        llm_provider = None

    # Handle deprecated llm_model_name keyword argument
    if llm_model_name := kwargs.pop("llm_model_name", None):
        warnings.warn(
            "llm_model_name is deprecated and will be removed in a future version. Use "
            "'default_model' instead.",
            stacklevel=2,
            category=DeprecationWarning,
        )

    legacy_model_kwargs = {}
    for legacy_model_key, new_model_key in {
        PLANNING_MODEL_KEY: "planning_model",
        EXECUTION_MODEL_KEY: "execution_model",
        INTROSPECTION_MODEL_KEY: "introspection_model",
        SUMMARISER_MODEL_KEY: "summarizer_model",
        DEFAULT_MODEL_KEY: "default_model",
    }.items():
        if legacy_model_key in kwargs:
            warnings.warn(
                f"{legacy_model_key} is deprecated and will be removed in a future version. Use "
                f"{new_model_key} instead.",
                stacklevel=2,
                category=DeprecationWarning,
            )
            legacy_model_kwargs[new_model_key] = kwargs.pop(legacy_model_key)

    models = kwargs.pop("models", {})
    if isinstance(models, GenerativeModelsConfig):
        models = models.model_dump(exclude_unset=True)
    duplicate_model_keys = kwargs.keys() & models.keys()
    if duplicate_model_keys:
        raise InvalidConfigError(
            ", ".join(duplicate_model_keys),
            "Model passed in Keys in kwargs and models must be unique",
        )

    def filter_none(mapping: dict[str, Any]) -> dict[str, Any]:
        """Filter out None values from a dictionary."""
        return {k: v for k, v in mapping.items() if v is not None}

    kwargs_models = {
        **filter_none(legacy_model_kwargs),
        **filter_none({"default_model": llm_model_name}),
        **filter_none(models),
        **filter_none(
            {k: v for k, v in kwargs.items() if k in GenerativeModelsConfig.model_fields},
        ),
    }

    models = GenerativeModelsConfig(
        default_model=kwargs_models.get("default_model"),
        planning_model=kwargs_models.get("planning_model"),
        execution_model=kwargs_models.get("execution_model"),
        introspection_model=kwargs_models.get("introspection_model"),
        summarizer_model=kwargs_models.get("summarizer_model"),
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
        execution_agent_type=kwargs.pop("execution_agent_type", ExecutionAgentType.ONE_SHOT),
        **kwargs,
    )
