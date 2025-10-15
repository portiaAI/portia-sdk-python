"""Integration tests for the CLI (template-aligned)."""

import os
import tempfile
import warnings
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import toml
from click.testing import CliRunner
from pydantic import SecretStr

from portia.cli import cli
from portia.config import Config, StorageClass, default_config
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
from portia.model import GenerativeModel, LLMProvider
from portia.open_source_tools.llm_tool import LLMTool
from portia.clarification_handler import ClarificationHandler # noqa: F401

@pytest.fixture
def temp_config_dir() -> Generator[Path, None, None]:
    """Create a temporary config directory and point loader defaults there."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / ".portia"
        config_dir.mkdir()
        with (
            patch.object(ConfigLoader, "DEFAULT_CONFIG_DIR", config_dir),
            patch.object(ConfigLoader, "DEFAULT_CONFIG_FILE", config_dir / "config.toml"),
        ):
            yield config_dir


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock required environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    # Use a plain string for endpoint to avoid SecretStr -> URL issues in http clients
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.azure.com")


@pytest.fixture
def mock_portia_cls() -> Generator[MagicMock, None, None]:
    """Mock the Portia class used by the CLI."""
    with patch("portia.cli.Portia", autospec=True) as mock_portia:
        yield mock_portia


@pytest.fixture(autouse=True)
def mock_config_get_generative_model() -> Generator[None, None, None]:
    """Avoid real model initialization inside CLI."""
    with patch.object(Config, "get_generative_model") as mock_get_generative_model:
        mock_get_generative_model.return_value = MagicMock(spec=GenerativeModel)
        yield None


def test_cli_run_basic(mock_portia_cls: MagicMock) -> None:
    """Test the CLI run command with confirmation."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "Calculate 1 + 2"], input="y\n")
    assert result.exit_code == 0

    mock_portia = mock_portia_cls.return_value
    assert mock_portia.plan.call_count == 1
    assert mock_portia.plan.call_args[0][0] == "Calculate 1 + 2"
    assert mock_portia.run_plan.call_count == 1
    assert mock_portia.run_plan.call_args[0][0] is mock_portia.plan.return_value


@pytest.mark.parametrize(
    ("provider", "expected_provider"),
    [
        ("anthropic", LLMProvider.ANTHROPIC),
        ("google", LLMProvider.GOOGLE),
        ("azure-openai", LLMProvider.AZURE_OPENAI),
        ("mistralai", LLMProvider.MISTRALAI),
        ("openai", LLMProvider.OPENAI),
    ],
)
def test_cli_run_config_set_provider(
    mock_portia_cls: MagicMock, provider: str, expected_provider: LLMProvider
) -> None:
    """Test --llm-provider mapping to Config.llm_provider enum."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "Calculate 1 + 2", "--llm-provider", provider], input="y\n")
    assert result.exit_code == 0
    assert mock_portia_cls.call_count == 1
    cfg = mock_portia_cls.call_args.kwargs["config"]
    assert cfg.llm_provider == expected_provider


def test_cli_run_config_set_planner_model(mock_portia_cls: MagicMock) -> None:
    """Test --planning-model overrides planning model in config."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "Calculate 1 + 2", "--planning-model", "openai/gpt-3.5-turbo"],
        input="y\n",
    )
    assert result.exit_code == 0
    assert mock_portia_cls.call_count == 1
    cfg = mock_portia_cls.call_args.kwargs["config"]
    assert cfg.models.planning_model == "openai/gpt-3.5-turbo"


def test_cli_run_config_multi_setting(mock_portia_cls: MagicMock) -> None:
    """Test multiple config options and tool selection in a single run."""
    with patch("portia.cli.DefaultToolRegistry", autospec=True) as mock_tool_registry:
        llm_tool = LLMTool()
        mock_tool_registry.return_value.get_tools.return_value = [llm_tool]
        mock_tool_registry.return_value.match_tools.return_value = [llm_tool]

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "Calculate 1 + 2",
                "--planning-model",
                "openai/gpt-3.5-turbo",
                "--llm-provider",
                "anthropic",
                "--storage-class",
                "MEMORY",
                "--tool-id",
                "llm_tool",
            ],
            input="y\n",
        )

    assert result.exit_code == 0
    assert mock_portia_cls.call_count == 1
    cfg = mock_portia_cls.call_args.kwargs["config"]
    assert cfg.models.planning_model == "openai/gpt-3.5-turbo"
    assert cfg.llm_provider == LLMProvider.ANTHROPIC
    assert cfg.storage_class == StorageClass.MEMORY
    tools = mock_portia_cls.call_args.kwargs["tools"]
    assert len(tools) == 1
    assert tools[0].id == "llm_tool"


def test_cli_run_no_confirmation(mock_portia_cls: MagicMock) -> None:
    """Test running without confirmation prompt."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "Compute 3 * 3", "--confirm", "false"])
    assert result.exit_code == 0

    mock_portia = mock_portia_cls.return_value
    assert mock_portia.plan.call_count == 1
    assert mock_portia.plan.call_args[0][0] == "Compute 3 * 3"
    assert mock_portia.run_plan.call_count == 1


def test_cli_run_custom_end_user_id(mock_portia_cls: MagicMock) -> None:
    """Test passing a custom end user id through CLI."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "Sum 1 + 1", "--end-user-id", "user-123", "--confirm", "false"],
    )
    assert result.exit_code == 0

    mock_portia = mock_portia_cls.return_value
    assert mock_portia.plan.call_args[0][0] == "Sum 1 + 1"
    assert mock_portia.plan.call_args[1]["end_user"] == "user-123"
    assert mock_portia.run_plan.call_count == 1


def test_cli_run_reject_confirmation(mock_portia_cls: MagicMock) -> None:
    """Test rejecting plan execution at the prompt."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "Subtract 5 - 3"], input="n\n")
    assert result.exit_code == 0

    mock_portia = mock_portia_cls.return_value
    assert mock_portia.plan.call_count == 1
    assert mock_portia.run_plan.call_count == 0


def test_cli_plan_default(mock_portia_cls: MagicMock) -> None:
    """Test the CLI plan command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["plan", "What is the weather?"])
    assert result.exit_code == 0

    mock_portia = mock_portia_cls.return_value
    assert mock_portia.plan.call_count == 1
    assert mock_portia.plan.call_args[0][0] == "What is the weather?"


def test_cli_list_tools() -> None:
    """Test listing available tools."""
    llm_tool = LLMTool()
    with patch("portia.cli.DefaultToolRegistry", autospec=True) as mock_tool_registry:
        mock_tool_registry.return_value.get_tools.return_value = [llm_tool]
        runner = CliRunner()
        result = runner.invoke(cli, ["list-tools"])
    assert result.exit_code == 0
    assert llm_tool.pretty() in result.output


def test_cli_version() -> None:
    """Test printing CLI version."""
    runner = CliRunner()
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    # version like 1.2.3 or 1.2.3-alpha
    import re

    assert re.match(r"\d+\.\d+\.\d+-?\w*", result.output) is not None


def test_profile_precedence(temp_config_dir: Path) -> None:
    """Test that a selected profile's configuration is passed through correctly."""
    # Create a test profile in the temp config
    config_file = temp_config_dir / "config.toml"
    data = {
        "profile": {
            "test": {
                "llm_provider": "openai",
                "default_model": "openai/gpt-4o",
                "storage_class": "CLOUD",
            }
        }
    }
    with Path.open(config_file, "w") as f:
        toml.dump(data, f)

    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        patch("portia.cli._get_config") as mock_get_config,
    ):
        mock_cli_config = Mock()
        mock_config = Mock()

        def must_get_side_effect(*args: object, **_kwargs: object) -> object:
            key = args[0] if args else None
            if key and "key" in str(key).lower():
                return SecretStr("test-api-key")
            if key and ("endpoint" in str(key).lower() or "url" in str(key).lower()):
                return "https://api.portialabs.ai"
            return "some-value"

        mock_config.must_get.side_effect = must_get_side_effect
        mock_config.must_get_api_key.return_value = SecretStr("test-api-key")
        mock_config.get_default_model.return_value = Mock()

        mock_get_config.return_value = (mock_cli_config, mock_config)

        with patch("portia.tool_registry.DefaultToolRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_tool = Mock()
            mock_tool.pretty.return_value = "Mock Tool\nDescription: A test tool"
            mock_registry.get_tools.return_value = [mock_tool]

            with patch("httpx.Client.get") as mock_httpx_get:
                mock_httpx_get.return_value = Mock(status_code=200, json=lambda: {"tools": []})

                runner = CliRunner()
                result = runner.invoke(cli, ["--profile", "test", "list-tools"])

                assert result.exit_code == 0
                mock_get_config.assert_called_once()
                call_kwargs = mock_get_config.call_args[1]
                assert call_kwargs["profile"] == "test"


def test_cli_config_create_and_list() -> None:
    """Test creating and listing config profiles."""
    runner = CliRunner()

    result = runner.invoke(cli, ["config", "create", "myprofile"])
    assert result.exit_code == 0

    result = runner.invoke(cli, ["config", "list"])
    assert "myprofile" in result.output


def test_cli_config_set_and_get() -> None:
    """Test setting and getting a config value."""
    runner = CliRunner()

    runner.invoke(cli, ["config", "create", "myprofile"])

    result = runner.invoke(cli, ["config", "set", "myprofile", "llm_provider=openai"])
    assert result.exit_code == 0

    result = runner.invoke(cli, ["config", "get", "myprofile", "llm_provider"])
    assert "openai" in result.output


def test_cli_config_set_default_and_get() -> None:
    """Test setting and getting the default profile."""
    runner = CliRunner()
    runner.invoke(cli, ["config", "create", "foo"])
    runner.invoke(cli, ["config", "create", "bar"])

    result = runner.invoke(cli, ["config", "set-default", "bar"])
    assert result.exit_code == 0

    result = runner.invoke(cli, ["config", "get", "bar"])
    assert result.exit_code == 0


def test_cli_config_delete() -> None:
    """Test config delete command."""
    runner = CliRunner()
    runner.invoke(cli, ["config", "create", "todelete"])

    result = runner.invoke(cli, ["config", "delete", "todelete"], input="y\n")
    assert result.exit_code == 0
    result = runner.invoke(cli, ["config", "list"])
    assert "todelete" not in result.output


def test_cli_config_path() -> None:
    """Test config path command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["config", "path"])
    assert result.exit_code == 0
    assert ".portia" in result.output


def test_cli_config_validate() -> None:
    """Test config validation command."""
    runner = CliRunner()
    runner.invoke(cli, ["config", "create", "validprofile"])
    result = runner.invoke(cli, ["config", "validate", "validprofile"])
    assert result.exit_code == 0


def test_config_loader_missing_config_file(temp_config_dir: Path) -> None:
    """Test behavior when config file doesn't exist."""
    loader = ConfigLoader(temp_config_dir / "nonexistent.toml")

    with pytest.raises(ConfigNotFoundError):
        loader.load_config_from_toml("default")


def test_config_loader_invalid_toml(temp_config_dir: Path) -> None:
    """Test behavior with malformed TOML file."""
    config_file = temp_config_dir / "config.toml"
    with Path.open(config_file, "w") as f:
        f.write("invalid toml content [[[")

    loader = ConfigLoader(config_file)

    with pytest.raises(InvalidConfigError):
        loader.load_config_from_toml("default")


def test_config_loader_missing_profile(temp_config_dir: Path) -> None:
    """Test behavior when requested profile doesn't exist."""
    config_file = temp_config_dir / "config.toml"
    data = {"profile": {"default": {"llm_provider": "openai"}}}
    with Path.open(config_file, "w") as f:
        toml.dump(data, f)

    loader = ConfigLoader(config_file)

    with pytest.raises(ConfigNotFoundError):
        loader.load_config_from_toml("nonexistent")


def test_config_env_var_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test environment variable overrides."""
    monkeypatch.setenv("PORTIA_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("PORTIA_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("PORTIA_JSON_LOG_SERIALIZE", "true")
    monkeypatch.setenv("PORTIA_LARGE_OUTPUT_THRESHOLD_TOKENS", "2000")

    loader = ConfigLoader()
    config = loader.merge_with_env({})

    assert config["llm_provider"] == "anthropic"
    assert config["default_log_level"] == "DEBUG"
    assert config["json_log_serialize"] is True
    assert config["large_output_threshold_tokens"] == 2000


def test_config_feature_flags_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test feature flags from environment variables."""
    monkeypatch.setenv("PORTIA_FEATURE_test_flag", "true")
    monkeypatch.setenv("PORTIA_FEATURE_another_flag", "false")

    loader = ConfigLoader()
    config = loader.merge_with_env({})

    assert config["feature_flags"]["test_flag"] is True
    assert config["feature_flags"]["another_flag"] is False


def test_config_loader_list_profiles_no_file(temp_config_dir: Path) -> None:
    """Test listing profiles when no config file exists."""
    loader = ConfigLoader(temp_config_dir / "nonexistent.toml")
    profiles = loader.list_profiles()

    assert profiles == []


def test_config_loader_list_profiles_invalid_file(temp_config_dir: Path) -> None:
    """Test listing profiles with invalid config file."""
    # Create invalid file
    config_file = temp_config_dir / "config.toml"
    with Path.open(config_file, "w") as f:
        f.write("invalid content")

    loader = ConfigLoader(config_file)
    profiles = loader.list_profiles()

    assert profiles == []


def test_config_default_profile_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test default profile from environment variable."""
    monkeypatch.setenv("PORTIA_DEFAULT_PROFILE", "custom")

    loader = ConfigLoader()
    default_profile = loader.get_default_profile()

    assert default_profile == "custom"


def test_config_integer_env_var_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test invalid integer environment variable."""
    monkeypatch.setenv("PORTIA_LARGE_OUTPUT_THRESHOLD_TOKENS", "not_a_number")

    loader = ConfigLoader()
    config = loader.merge_with_env({})
    assert "large_output_threshold_tokens" not in config


def test_config_from_local_config_no_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Config.from_local_config when no file exists."""
    monkeypatch.setenv("PORTIA_LLM_PROVIDER", "openai")
    config = Config.from_local_config(profile="nonexistent")
    assert config is not None


def test_default_config_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test warnings in default_config function."""
    # Clear all API keys to trigger warning
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    with pytest.raises(InvalidConfigError):
        default_config()


def test_default_config_deprecated_args() -> None:
    """Test deprecated arguments in default_config."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        assert len(w) > 0
        assert "llm_model_name is deprecated" in str(w[0].message)

        # Test deprecated model keys
        assert any("planning_model_name is deprecated" in str(warning.message) for warning in w)


def test_config_storage_class_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test storage class default logic."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PORTIA_LLM_PROVIDER", "openai")
    config = Config.from_local_config(profile="default")
    assert config.storage_class in (StorageClass.CLOUD, StorageClass.MEMORY)


def test_config_must_get_errors() -> None:
    """Test Config.must_get error cases."""
    config = Config.from_default(openai_api_key="test")

    # Test non-existent attribute
    with pytest.raises(ConfigNotFoundError):
        config.must_get("nonexistent_field", str)

    # Test wrong type
    with pytest.raises(InvalidConfigError):
        config.must_get("openai_api_key", int)  # SecretStr, not int


def test_config_empty_values() -> None:
    """Test Config.must_get with empty values."""
    config = Config.from_default(
        openai_api_key="test",
        azure_openai_endpoint="",  # Empty string
    )

    # Test empty string
    with pytest.raises(InvalidConfigError):
        config.must_get("azure_openai_endpoint", str)

    # Test empty SecretStr
    config.openai_api_key = SecretStr("")
    with pytest.raises(InvalidConfigError):
        config.must_get("openai_api_key", SecretStr)


def test_construct_model_from_name_custom_provider_raises() -> None:
    """Test that constructing model from name with CUSTOM provider raises error."""
    config = Config.from_default(openai_api_key="test")
    with pytest.raises(ValueError, match="Cannot construct a custom model from a string"):
        config._construct_model_from_name(LLMProvider.CUSTOM, "my-custom-model")


def test_list_profiles_handles_exception(temp_config_dir: Path) -> None:
    """Test that list_profiles handles exceptions gracefully."""
    config_file = temp_config_dir / "config.toml"
    with Path.open(config_file, "w") as f:
        f.write("not a toml file")
    loader = ConfigLoader(config_file)
    assert loader.list_profiles() == []


def test_ensure_config_directory_and_get_config_file_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test ensure_config_directory and get_config_file_path helpers."""
    monkeypatch.setattr(ConfigLoader, "DEFAULT_CONFIG_DIR", tmp_path / ".portia")
    monkeypatch.setattr(ConfigLoader, "DEFAULT_CONFIG_FILE", tmp_path / ".portia" / "config.toml")
    config_dir = ensure_config_directory()
    assert config_dir.exists()
    assert config_dir.name == ".portia"
    config_file = get_config_file_path()
    assert config_file.name == "config.toml"


def test_list_profiles_happy_path(temp_config_dir: Path) -> None:
    """Test listing profiles from a valid config file."""
    cfg_toml = temp_config_dir / "config.toml"
    data = {"profile": {"one": {}, "two": {}}}
    with Path.open(cfg_toml, "w") as f:
        toml.dump(data, f)

    loader = ConfigLoader(cfg_toml)
    profiles = loader.list_profiles()
    assert sorted(profiles) == ["one", "two"]


def test_helper_load_config_from_toml(temp_config_dir: Path) -> None:
    """Test the load_config_from_toml helper function."""
    data = {"profile": {"foo": {"llm_provider": "openai"}}}
    cfg_file = temp_config_dir / "config.toml"
    with Path.open(cfg_file, "w") as f:
        toml.dump(data, f)

    result = load_config_from_toml("foo", config_file=cfg_file)
    assert result["llm_provider"] == "openai"


def test_helper_merge_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the merge_with_env helper function."""
    monkeypatch.setenv("PORTIA_LLM_PROVIDER", "anthropic")
    merged = merge_with_env({"some_key": "value"})
    assert merged["llm_provider"] == "anthropic"
    assert merged["some_key"] == "value"


def test_helper_apply_overrides() -> None:
    """Test the apply_overrides helper function."""
    base = {"a": 1, "feature_flags": {"x": True}}
    overrides = {"a": 2, "feature_flags": {"y": False}}
    out = apply_overrides(base, overrides)
    assert out["a"] == 2
    assert out["feature_flags"] == {"x": True, "y": False}


def test_helper_get_config_precedence(
    temp_config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test get_config precedence: direct args > env vars > config file."""
    # create a profile in TOML
    data = {"profile": {"bar": {"llm_provider": "openai"}}}
    cfg_file = temp_config_dir / "config.toml"
    with Path.open(cfg_file, "w") as f:
        toml.dump(data, f)
    # set env var that should be overridden by code override
    monkeypatch.setenv("PORTIA_LLM_PROVIDER", "anthropic")
    # direct override to azure
    result = get_config("bar", config_file=cfg_file, llm_provider="azure-openai")
    assert result["llm_provider"] == "azure-openai"


def test_helper_get_config_no_file_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_config falls back to env vars if no config file."""
    monkeypatch.setenv("PORTIA_LLM_PROVIDER", "google")
    cfg = get_config("any", config_file=Path("/does/not/exist.toml"))
    assert cfg["llm_provider"] == "google"


def test_ensure_config_directory_and_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ensure_config_directory and get_config_file_path helpers."""
    monkeypatch.setattr(ConfigLoader, "DEFAULT_CONFIG_DIR", tmp_path / ".portia")
    monkeypatch.setattr(ConfigLoader, "DEFAULT_CONFIG_FILE", tmp_path / ".portia" / "config.toml")

    cfg_dir = ensure_config_directory()
    assert cfg_dir.exists()
    assert cfg_dir.name == ".portia"

    path = get_config_file_path()
    assert path.name == "config.toml"


def test_fill_default_models_sets_planning_and_introspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that default planning and introspection models are set based on LLM provider."""
    from portia.config import Config, GenerativeModelsConfig, LLMProvider

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    models = GenerativeModelsConfig(default_model="openai/gpt-4.1")
    c = Config.from_default(
        llm_provider=LLMProvider.OPENAI, models=models, openai_api_key="test-key"
    )
    assert c.models.planning_model == "openai/o3-mini"
    assert c.models.introspection_model == "openai/o4-mini"


def test_config_loader_wrappers(temp_config_dir: Path) -> None:
    """Test the load_config_from_toml and get_config helper functions."""
    cfg_file = temp_config_dir / "config.toml"
    data = {"profile": {"foo": {"llm_provider": "openai"}}}
    with Path.open(cfg_file, "w") as f:
        toml.dump(data, f)
    result = load_config_from_toml("foo", config_file=cfg_file)
    assert result["llm_provider"] == "openai"
    result2 = get_config("foo", config_file=cfg_file)
    assert result2["llm_provider"] == "openai"
