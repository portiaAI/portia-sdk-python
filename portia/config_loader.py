"""Configuration loader for TOML-based profile system.

This module provides functionality to load configuration from TOML files,
handle profile selection, and merge settings with proper precedence:
1. Direct code overrides (highest)
2. Config file values
3. Environment variables (lowest)
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, ClassVar

from portia.errors import ConfigNotFoundError, InvalidConfigError


class ConfigLoader:
    """Handles loading and merging of TOML configuration files with profiles."""

    DEFAULT_CONFIG_DIR = Path.home() / ".portia"
    DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.toml"

    ENV_VAR_MAPPING: ClassVar[dict[str, str]] = {
        # Portia Cloud
        "portia_api_endpoint": "PORTIA_API_ENDPOINT",
        "portia_dashboard_url": "PORTIA_DASHBOARD_URL",
        "portia_api_key": "PORTIA_API_KEY",
        # LLM API Keys
        "openrouter_api_key": "OPENROUTER_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "mistralai_api_key": "MISTRAL_API_KEY",
        "google_api_key": "GOOGLE_API_KEY",
        "azure_openai_api_key": "AZURE_OPENAI_API_KEY",
        "azure_openai_endpoint": "AZURE_OPENAI_ENDPOINT",
        "ollama_base_url": "OLLAMA_BASE_URL",
        # AWS
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "aws_default_region": "AWS_DEFAULT_REGION",
        "aws_credentials_profile_name": "AWS_CREDENTIALS_PROFILE_NAME",
        # Cache
        "llm_redis_cache_url": "LLM_REDIS_CACHE_URL",
        # Storage
        "storage_class": "PORTIA_STORAGE_CLASS",
        "storage_dir": "PORTIA_STORAGE_DIR",
        # Logging
        "default_log_level": "PORTIA_LOG_LEVEL",
        "default_log_sink": "PORTIA_LOG_SINK",
        "json_log_serialize": "PORTIA_JSON_LOG_SERIALIZE",
        # Models
        "llm_provider": "PORTIA_LLM_PROVIDER",
        "default_model": "PORTIA_DEFAULT_MODEL",
        "planning_model": "PORTIA_PLANNING_MODEL",
        "execution_model": "PORTIA_EXECUTION_MODEL",
        "introspection_model": "PORTIA_INTROSPECTION_MODEL",
        "summarizer_model": "PORTIA_SUMMARIZER_MODEL",
        # Agent settings
        "execution_agent_type": "PORTIA_EXECUTION_AGENT_TYPE",
        "planning_agent_type": "PORTIA_PLANNING_AGENT_TYPE",
        "argument_clarifications_enabled": "PORTIA_ARGUMENT_CLARIFICATIONS_ENABLED",
        "large_output_threshold_tokens": "PORTIA_LARGE_OUTPUT_THRESHOLD_TOKENS",
    }

    def __init__(self, config_file: Path | None = None) -> None:
        """Initialize the config loader.

        Args:
            config_file: Optional path to config file. Defaults to ~/.portia/config.toml

        """
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE

    def load_config_from_toml(self, profile: str = "default") -> dict[str, Any]:
        """Load configuration from TOML file for the specified profile.

        Args:
            profile: Profile name to load (e.g., "default", "openai", "gemini")

        Returns:
            Dictionary containing the profile configuration

        Raises:
            ConfigNotFoundError: If config file or profile doesn't exist
            InvalidConfigError: If TOML file is malformed

        """
        if not self.config_file.exists():
            raise ConfigNotFoundError(
                f"Config file not found: {self.config_file}. "
                f"Run 'portia-cli config init' to create one."
            )

        try:
            with Path.open(self.config_file, "rb") as f:
                toml_data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise InvalidConfigError(
                "config_file", f"Invalid TOML syntax in {self.config_file}: {e}"
            ) from e
        except (InvalidConfigError, ConfigNotFoundError) as e:
            raise ConfigNotFoundError(f"Error reading config file {self.config_file}: {e}") from e

        profiles = toml_data.get("profile", {})
        if profile not in profiles:
            available_profiles = list(profiles.keys())
            raise ConfigNotFoundError(
                f"Profile '{profile}' not found in {self.config_file}. "
                f"Available profiles: {available_profiles}"
            )
        return profiles[profile].copy()

    def merge_with_env(self, config: dict[str, Any]) -> dict[str, Any]:
        """Fill missing values in config with environment variables.

        Args:
            config: Configuration dictionary from TOML file

        Returns:
            Configuration dictionary with environment variables merged in

        """
        merged_config = config.copy()

        def _merge_env_var(config_key: str, env_var: str) -> None:
            current_value = merged_config.get(config_key)
            if current_value in (None, "", []):
                env_value = os.getenv(env_var)
                if env_value:
                    merged_config[config_key] = self._parse_env_value(config_key, env_value)

        for config_key, env_var in self.ENV_VAR_MAPPING.items():
            _merge_env_var(config_key, env_var)

        self._merge_feature_flags(merged_config)
        return merged_config

    def _parse_env_value(self, config_key: str, env_value: str) -> bool | int | str:
        """Parse environment variable value based on config key."""
        if config_key in ["json_log_serialize", "argument_clarifications_enabled"]:
            return env_value.lower() in ("true", "1", "yes", "on")
        if config_key in ["large_output_threshold_tokens"]:
            import contextlib

            with contextlib.suppress(ValueError):
                return int(env_value)
            return env_value  # fallback if int conversion fails
        return env_value

    def _merge_feature_flags(self, merged_config: dict[str, Any]) -> None:
        """Merge PORTIA_FEATURE_* env vars into feature_flags."""
        feature_flags = merged_config.get("feature_flags", {})
        for env_key, env_value in os.environ.items():
            if env_key.startswith("PORTIA_FEATURE_"):
                flag_name = env_key.lower().replace("portia_feature_", "")
                if flag_name not in feature_flags:
                    feature_flags[flag_name] = env_value.lower() in ("true", "1", "yes", "on")
        if feature_flags:
            merged_config["feature_flags"] = feature_flags

    def apply_overrides(self, config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        """Apply code-based overrides to the configuration.

        Args:
            config: Base configuration dictionary
            overrides: Override values from code

        Returns:
            Configuration with overrides applied

        """
        final_config = config.copy()

        for key, value in overrides.items():
            if value is not None:
                if key == "feature_flags" and isinstance(value, dict):
                    existing_flags = final_config.get("feature_flags", {})
                    existing_flags.update(value)
                    final_config["feature_flags"] = existing_flags
                else:
                    final_config[key] = value

        return final_config

    def get_config(self, profile: str = "default", **overrides: Any) -> dict[str, Any]:
        """Load complete configuration with proper precedence.

        Precedence order (highest to lowest):
        1. Direct code overrides (**overrides)
        2. Config file values
        3. Environment variables

        Args:
            profile: Profile name to load
            **overrides: Direct overrides from code

        Returns:
            Final merged configuration dictionary

        """
        base_config = {}

        env_config = self.merge_with_env(base_config)

        try:
            file_config = self.load_config_from_toml(profile)

            merged_config = {**env_config, **file_config}

            if "feature_flags" in env_config and "feature_flags" in file_config:
                merged_feature_flags = {
                    **env_config.get("feature_flags", {}),
                    **file_config.get("feature_flags", {}),
                }
                merged_config["feature_flags"] = merged_feature_flags

        except ConfigNotFoundError:
            merged_config = env_config

        return self.apply_overrides(merged_config, overrides)

    def list_profiles(self) -> list[str]:
        """List all available profiles in the config file.

        Returns:
            List of profile names

        """
        if not self.config_file.exists():
            return []

        try:
            with Path.open(self.config_file, "rb") as f:
                toml_data = tomllib.load(f)
            return list(toml_data.get("profile", {}).keys())
        except (OSError, tomllib.TOMLDecodeError):
            return []

    def get_default_profile(self) -> str:
        """Get the default profile name.

        Returns:
            Default profile name, with fallback to "default"

        """
        return os.getenv("PORTIA_DEFAULT_PROFILE", "default")


def load_config_from_toml(
    profile: str = "default", config_file: Path | None = None
) -> dict[str, Any]:
    """Load configuration from TOML file for the specified profile.

    Args:
        profile: Profile name to load
        config_file: Optional path to config file

    Returns:
        Profile configuration dictionary

    """
    loader = ConfigLoader(config_file)
    return loader.load_config_from_toml(profile)


def merge_with_env(config: dict[str, Any]) -> dict[str, Any]:
    """Fill missing values in config with environment variables.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with environment variables merged

    """
    loader = ConfigLoader()
    return loader.merge_with_env(config)


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply code-based overrides to configuration.

    Args:
        config: Base configuration
        overrides: Override values

    Returns:
        Configuration with overrides applied

    """
    loader = ConfigLoader()
    return loader.apply_overrides(config, overrides)


def get_config(
    profile: str = "default", config_file: Path | None = None, **overrides: Any
) -> dict[str, Any]:
    """Load complete configuration with proper precedence.

    Args:
        profile: Profile name to load
        config_file: Optional path to config file
        **overrides: Direct overrides from code

    Returns:
        Final merged configuration dictionary

    """
    loader = ConfigLoader(config_file)
    return loader.get_config(profile, **overrides)


def ensure_config_directory() -> Path:
    """Ensure the config directory exists and return its path.

    Returns:
        Path to the config directory

    """
    config_dir = ConfigLoader.DEFAULT_CONFIG_DIR
    config_dir.mkdir(exist_ok=True, mode=0o755)
    return config_dir


def get_config_file_path() -> Path:
    """Get the path to the config file.

    Returns:
        Path to the config file

    """
    return ConfigLoader.DEFAULT_CONFIG_FILE
