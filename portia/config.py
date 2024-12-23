"""Configuration for the SDK."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, SecretStr

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


class AgentType(Enum):
    """Type of agent to use for executing a step."""

    TOOL_LESS = "TOOL_LESS"
    ONE_SHOT = "ONE_SHOT"
    VERIFIER = "VERIFIER"


class Config(BaseModel):
    """General configuration for the library."""

    # API Keys
    portia_api_key: SecretStr | None = SecretStr(os.getenv("PORTIA_API_KEY") or "")
    openai_api_key: SecretStr | None = SecretStr(os.getenv("OPENAI_API_KEY") or "")
    anthropic_api_key: SecretStr | None = SecretStr(os.getenv("ANTHROPIC_API_KEY") or "")
    mistralai_api_key: SecretStr | None = SecretStr(os.getenv("MISTRAL_API_KEY") or "")

    # Storage Options
    storage_class: StorageClass
    storage_dir: str | None = None

    # Registry Options
    enable_cloud_tool_registry: bool

    # LLM Options
    llm_provider: LLMProvider
    llm_model_name: str
    llm_model_temperature: int
    llm_model_seed: int

    # Agent Options
    default_agent_type: AgentType

    # System Context Overrides
    planner_system_context_override: list[str] | None = None
    agent_system_context_override: list[str] | None = None

    @classmethod
    def from_file(cls, file_path: Path) -> Config:
        """Load configuration from a JSON file."""
        with Path.open(file_path) as f:
            return cls.model_validate_json(f.read())

    def has_api_key(self, name: str) -> bool:
        """Check if the given API Key is available."""
        return hasattr(self, name)

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
            raise InvalidConfigError(name)
        return value


def default_config() -> Config:
    """Return default config."""
    return Config(
        storage_class=StorageClass.MEMORY,
        llm_provider=LLMProvider.OPENAI,
        llm_model_name="gpt-4o-mini",
        llm_model_temperature=0,
        llm_model_seed=443,
        default_agent_type=AgentType.VERIFIER,
        enable_cloud_tool_registry=False,
    )
