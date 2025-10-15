"""Unit tests for config_loader.py."""

from pathlib import Path
from unittest.mock import patch

import pytest

from portia.config_loader import (
    ConfigLoader,
    apply_overrides,
    ensure_config_directory,
    get_config,
    get_config_file_path,
    load_config_from_toml,
    merge_with_env,
)
from portia.errors import ConfigNotFoundError, InvalidConfigError

"""Test ConfigLoader class functionality."""


def test_init_default_config_file() -> None:
    """Test ConfigLoader initialization with default config file."""
    loader = ConfigLoader()
    assert loader.config_file == ConfigLoader.DEFAULT_CONFIG_FILE


def test_init_custom_config_file() -> None:
    """Test ConfigLoader initialization with custom config file."""
    custom_path = Path("/custom/config.toml")
    loader = ConfigLoader(custom_path)
    assert loader.config_file == custom_path


def test_load_config_from_toml_file_not_found() -> None:
    """Test load_config_from_toml raises ConfigNotFoundError when file doesn't exist."""
    loader = ConfigLoader(Path("/nonexistent/config.toml"))
    with pytest.raises(ConfigNotFoundError, match="Config file not found"):
        loader.load_config_from_toml()


def test_load_config_from_toml_invalid_toml(tmp_path: Path) -> None:
    """Test load_config_from_toml raises InvalidConfigError for malformed TOML."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("invalid toml content [[[")

    loader = ConfigLoader(config_file)
    with pytest.raises(InvalidConfigError, match="Invalid TOML syntax"):
        loader.load_config_from_toml()


def test_load_config_from_toml_profile_not_found(tmp_path: Path) -> None:
    """Test load_config_from_toml raises ConfigNotFoundError when profile doesn't exist."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[profile.default]
key = "value"
""")

    loader = ConfigLoader(config_file)
    with pytest.raises(ConfigNotFoundError, match="Profile 'nonexistent' not found"):
        loader.load_config_from_toml("nonexistent")


def test_load_config_from_toml_success(tmp_path: Path) -> None:
    """Test successful loading of config from TOML."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[profile.default]
openai_api_key = "test-key"
llm_provider = "openai"
""")

    loader = ConfigLoader(config_file)
    config = loader.load_config_from_toml("default")

    assert config["openai_api_key"] == "test-key"
    assert config["llm_provider"] == "openai"


def test_merge_with_env_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test merge_with_env with basic environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")
    monkeypatch.setenv("PORTIA_LOG_LEVEL", "DEBUG")

    loader = ConfigLoader()
    config = {"some_key": "some_value"}
    merged = loader.merge_with_env(config)

    assert merged["openai_api_key"] == "env-api-key"
    assert merged["default_log_level"] == "DEBUG"
    assert merged["some_key"] == "some_value"


def test_merge_with_env_boolean_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test merge_with_env converts boolean environment variables correctly."""
    monkeypatch.setenv("PORTIA_JSON_LOG_SERIALIZE", "true")
    monkeypatch.setenv("PORTIA_ARGUMENT_CLARIFICATIONS_ENABLED", "1")

    loader = ConfigLoader()
    config = {}
    merged = loader.merge_with_env(config)

    assert merged["json_log_serialize"] is True
    assert merged["argument_clarifications_enabled"] is True


def test_merge_with_env_integer_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test merge_with_env converts integer environment variables correctly."""
    monkeypatch.setenv("PORTIA_LARGE_OUTPUT_THRESHOLD_TOKENS", "5000")

    loader = ConfigLoader()
    config = {}
    merged = loader.merge_with_env(config)

    assert merged["large_output_threshold_tokens"] == 5000


def test_merge_with_env_integer_conversion_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test merge_with_env handles invalid integer conversion gracefully."""
    monkeypatch.setenv("PORTIA_LARGE_OUTPUT_THRESHOLD_TOKENS", "not_a_number")

    loader = ConfigLoader()
    config = {}
    merged = loader.merge_with_env(config)
    assert "large_output_threshold_tokens" not in merged


def test_merge_with_env_feature_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test merge_with_env handles PORTIA_FEATURE_ environment variables - COVERS LINES 111-112."""
    monkeypatch.setenv("PORTIA_FEATURE_experimental_mode", "true")
    monkeypatch.setenv("PORTIA_FEATURE_debug_enabled", "1")
    monkeypatch.setenv("PORTIA_FEATURE_beta_features", "false")

    loader = ConfigLoader()
    config = {}
    merged = loader.merge_with_env(config)

    assert merged["feature_flags"]["experimental_mode"] is True
    assert merged["feature_flags"]["debug_enabled"] is True
    assert merged["feature_flags"]["beta_features"] is False


def test_merge_with_env_existing_feature_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test merge_with_env with existing feature flags."""
    monkeypatch.setenv("PORTIA_FEATURE_new_feature", "true")

    loader = ConfigLoader()
    config = {"feature_flags": {"existing_feature": True}}
    merged = loader.merge_with_env(config)

    assert merged["feature_flags"]["existing_feature"] is True
    assert merged["feature_flags"]["new_feature"] is True


def test_merge_with_env_no_override_existing_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test merge_with_env doesn't override existing non-empty values."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    loader = ConfigLoader()
    config = {"openai_api_key": "existing-key"}
    merged = loader.merge_with_env(config)
    assert merged["openai_api_key"] == "existing-key"


def test_apply_overrides_basic() -> None:
    """Test apply_overrides with basic overrides."""
    loader = ConfigLoader()
    config = {"key1": "value1", "key2": "value2"}
    overrides = {"key2": "new_value", "key3": "value3"}

    result = loader.apply_overrides(config, overrides)

    assert result["key1"] == "value1"
    assert result["key2"] == "new_value"
    assert result["key3"] == "value3"


def test_apply_overrides_feature_flags() -> None:
    """Test apply_overrides with feature flags merging."""
    loader = ConfigLoader()
    config = {"feature_flags": {"flag1": True, "flag2": False}}
    overrides = {"feature_flags": {"flag2": True, "flag3": True}}

    result = loader.apply_overrides(config, overrides)

    assert result["feature_flags"]["flag1"] is True
    assert result["feature_flags"]["flag2"] is True
    assert result["feature_flags"]["flag3"] is True


def test_apply_overrides_none_values_ignored() -> None:
    """Test apply_overrides ignores None values."""
    loader = ConfigLoader()
    config = {"key1": "value1"}
    overrides = {"key1": None, "key2": "value2"}

    result = loader.apply_overrides(config, overrides)

    assert result["key1"] == "value1"
    assert result["key2"] == "value2"


def test_get_config_file_exists_with_feature_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test get_config when file exists with feature flags merging - COVERS LINES 220-222."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[profile.default]
openai_api_key = "file-key"

[profile.default.feature_flags]
file_feature = true
""")

    monkeypatch.setenv("PORTIA_FEATURE_env_feature", "true")

    loader = ConfigLoader(config_file)
    config = loader.get_config("default", custom_override="override_value")

    assert config["openai_api_key"] == "file-key"
    assert config["custom_override"] == "override_value"
    assert config["feature_flags"]["file_feature"] is True
    assert config["feature_flags"]["env_feature"] is True


def test_get_config_file_not_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_config when config file doesn't exist."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    loader = ConfigLoader(Path("/nonexistent/config.toml"))
    config = loader.get_config("default", override_key="override_value")

    assert config["openai_api_key"] == "env-key"
    assert config["override_key"] == "override_value"


def test_list_profiles_file_not_exists() -> None:
    """Test list_profiles when config file doesn't exist."""
    loader = ConfigLoader(Path("/nonexistent/config.toml"))
    profiles = loader.list_profiles()
    assert profiles == []


def test_list_profiles_success(tmp_path: Path) -> None:
    """Test list_profiles with existing config file."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[profile.default]
key = "value"

[profile.openai]
key = "value"

[profile.anthropic]
key = "value"
""")

    loader = ConfigLoader(config_file)
    profiles = loader.list_profiles()

    assert set(profiles) == {"default", "openai", "anthropic"}


def test_list_profiles_malformed_file(tmp_path: Path) -> None:
    """Test list_profiles with malformed config file."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("invalid toml [[[")

    loader = ConfigLoader(config_file)
    profiles = loader.list_profiles()
    assert profiles == []


def test_get_default_profile_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_default_profile with environment variable."""
    monkeypatch.setenv("PORTIA_DEFAULT_PROFILE", "custom_profile")

    loader = ConfigLoader()
    profile = loader.get_default_profile()
    assert profile == "custom_profile"


def test_get_default_profile_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_default_profile fallback to 'default'."""
    monkeypatch.delenv("PORTIA_DEFAULT_PROFILE", raising=False)

    loader = ConfigLoader()
    profile = loader.get_default_profile()
    assert profile == "default"


def test_load_config_from_toml_function(tmp_path: Path) -> None:
    """Test load_config_from_toml standalone function."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[profile.test]
key = "value"
""")

    config = load_config_from_toml("test", config_file)
    assert config["key"] == "value"


def test_merge_with_env_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test merge_with_env standalone function."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = {"existing": "value"}
    merged = merge_with_env(config)

    assert merged["existing"] == "value"
    assert merged["openai_api_key"] == "test-key"


def test_apply_overrides_function() -> None:
    """Test apply_overrides standalone function."""
    config = {"key1": "value1"}
    overrides = {"key2": "value2"}

    result = apply_overrides(config, overrides)

    assert result["key1"] == "value1"
    assert result["key2"] == "value2"


def test_get_config_function(tmp_path: Path) -> None:
    """Test get_config standalone function."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[profile.default]
file_key = "file_value"
""")

    config = get_config("default", config_file, override_key="override_value")

    assert config["file_key"] == "file_value"
    assert config["override_key"] == "override_value"


def test_ensure_config_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ensure_config_directory creates directory."""
    test_dir = tmp_path / "test_portia"
    monkeypatch.setattr(ConfigLoader, "DEFAULT_CONFIG_DIR", test_dir)
    result_dir = ensure_config_directory()
    assert result_dir.exists()
    assert result_dir.is_dir()
    assert result_dir == test_dir


def test_get_config_file_path() -> None:
    """Test get_config_file_path returns correct path."""
    path = get_config_file_path()
    assert path == ConfigLoader.DEFAULT_CONFIG_FILE


def test_load_config_from_toml_handles_unexpected_exception(tmp_path: Path) -> None:
    """Test load_config_from_toml handles unexpected exceptions gracefully."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("[profile.default]\nkey='value'\n")
    loader = ConfigLoader(config_file)

    # Patch tomllib.load to raise an unexpected exception
    with (
        patch("tomllib.load", side_effect=RuntimeError("Unexpected error")),
        pytest.raises(ConfigNotFoundError, match="Unexpected error"),
    ):
        loader.load_config_from_toml()
