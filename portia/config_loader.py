"""Configuration loader for TOML-based profile system.

This module provides functionality to load configuration from TOML files,
handle profile selection, and merge settings with proper precedence:
1. Direct code overrides (highest)
2. Config file values  
3. Environment variables (lowest)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Handle Python version compatibility for TOML loading
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "tomli is required for Python < 3.11. Install with: pip install tomli"
        )

from portia.errors import ConfigNotFoundError, InvalidConfigError


class ConfigLoader:
    """Handles loading and merging of TOML configuration files with profiles."""
    
    DEFAULT_CONFIG_DIR = Path.home() / ".portia"
    DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.toml"
    
    # Environment variable mappings for all config fields
    ENV_VAR_MAPPING = {
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
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize the config loader.
        
        Args:
            config_file: Optional path to config file. Defaults to ~/.portia/config.toml
        """
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
    
    def load_config_from_toml(self, profile: str = "default") -> Dict[str, Any]:
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
            with open(self.config_file, "rb") as f:
                toml_data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise InvalidConfigError(
                "config_file", 
                f"Invalid TOML syntax in {self.config_file}: {e}"
            )
        except Exception as e:
            raise ConfigNotFoundError(f"Error reading config file {self.config_file}: {e}")
        
        # Extract profile data
        profiles = toml_data.get("profile", {})
        if profile not in profiles:
            available_profiles = list(profiles.keys())
            raise ConfigNotFoundError(
                f"Profile '{profile}' not found in {self.config_file}. "
                f"Available profiles: {available_profiles}"
            )
        
        profile_config = profiles[profile].copy()
        
        # Handle nested feature_flags section
        if "feature_flags" in profile_config:
            # Keep feature_flags as a nested dict
            pass
        
        return profile_config
    
    def merge_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing values in config with environment variables.
        
        Args:
            config: Configuration dictionary from TOML file
            
        Returns:
            Configuration dictionary with environment variables merged in
        """
        merged_config = config.copy()
        
        for config_key, env_var in self.ENV_VAR_MAPPING.items():
            # Only set from env var if not already present in config or if empty
            current_value = merged_config.get(config_key)
            if current_value in (None, "", []):
                env_value = os.getenv(env_var)
                if env_value:
                    # Handle boolean environment variables
                    if config_key in ["json_log_serialize", "argument_clarifications_enabled"]:
                        merged_config[config_key] = env_value.lower() in ("true", "1", "yes", "on")
                    # Handle integer environment variables  
                    elif config_key in ["large_output_threshold_tokens"]:
                        try:
                            merged_config[config_key] = int(env_value)
                        except ValueError:
                            pass  # Keep original value if conversion fails
                    else:
                        merged_config[config_key] = env_value
        
        # Handle special case for feature flags from environment
        feature_flags = merged_config.get("feature_flags", {})
        for env_key, env_value in os.environ.items():
            if env_key.startswith("PORTIA_FEATURE_"):
                flag_name = env_key.lower().replace("portia_feature_", "")
                if flag_name not in feature_flags:
                    feature_flags[flag_name] = env_value.lower() in ("true", "1", "yes", "on")
        
        if feature_flags:
            merged_config["feature_flags"] = feature_flags
            
        return merged_config
    
    def apply_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply code-based overrides to the configuration.
        
        Args:
            config: Base configuration dictionary
            overrides: Override values from code
            
        Returns:
            Configuration with overrides applied
        """
        final_config = config.copy()
        
        for key, value in overrides.items():
            if value is not None:  # Only override with non-None values
                # Handle nested feature_flags
                if key == "feature_flags" and isinstance(value, dict):
                    existing_flags = final_config.get("feature_flags", {})
                    existing_flags.update(value)
                    final_config["feature_flags"] = existing_flags
                else:
                    final_config[key] = value
        
        return final_config
    
    def get_config(self, profile: str = "default", **overrides) -> Dict[str, Any]:
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
        # Step 1: Start with empty config (env vars will be the base)
        base_config = {}
        
        # Step 2: Load and merge environment variables first (lowest precedence)
        env_config = self.merge_with_env(base_config)
        
        # Step 3: Load config file and merge (middle precedence)
        try:
            file_config = self.load_config_from_toml(profile)
            # Merge file config over env config
            merged_config = {**env_config, **file_config}
            
            # Handle nested merging for feature_flags
            if "feature_flags" in env_config and "feature_flags" in file_config:
                merged_feature_flags = {**env_config.get("feature_flags", {}), 
                                      **file_config.get("feature_flags", {})}
                merged_config["feature_flags"] = merged_feature_flags
                
        except ConfigNotFoundError:
            # If no config file exists, just use env config
            merged_config = env_config
        
        # Step 4: Apply code overrides (highest precedence)
        final_config = self.apply_overrides(merged_config, overrides)
        
        return final_config
    
    def list_profiles(self) -> list[str]:
        """List all available profiles in the config file.
        
        Returns:
            List of profile names
        """
        if not self.config_file.exists():
            return []
        
        try:
            with open(self.config_file, "rb") as f:
                toml_data = tomllib.load(f)
            return list(toml_data.get("profile", {}).keys())
        except Exception:
            return []
    
    def get_default_profile(self) -> str:
        """Get the default profile name.
        
        Returns:
            Default profile name, with fallback to "default"
        """
        # Could be extended to read from a separate config section
        # or environment variable like PORTIA_DEFAULT_PROFILE
        return os.getenv("PORTIA_DEFAULT_PROFILE", "default")


# Convenience functions for easy usage
def load_config_from_toml(profile: str = "default", config_file: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from TOML file for the specified profile.
    
    Args:
        profile: Profile name to load
        config_file: Optional path to config file
        
    Returns:
        Profile configuration dictionary
    """
    loader = ConfigLoader(config_file)
    return loader.load_config_from_toml(profile)


def merge_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing values in config with environment variables.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with environment variables merged
    """
    loader = ConfigLoader()
    return loader.merge_with_env(config)


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply code-based overrides to configuration.
    
    Args:
        config: Base configuration
        overrides: Override values
        
    Returns:
        Configuration with overrides applied
    """
    loader = ConfigLoader()
    return loader.apply_overrides(config, overrides)


def get_config(profile: str = "default", config_file: Optional[Path] = None, **overrides) -> Dict[str, Any]:
    """Main function to load complete configuration with proper precedence.
    
    Args:
        profile: Profile name to load
        config_file: Optional path to config file
        **overrides: Direct overrides from code
        
    Returns:
        Final merged configuration dictionary
    """
    loader = ConfigLoader(config_file)
    return loader.get_config(profile, **overrides)


# Utility function to create config directory
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
