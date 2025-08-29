"""Integration tests for the CLI (template-aligned)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import toml
from click.testing import CliRunner
from pydantic import SecretStr

from portia.cli import cli
from portia.config import Config, StorageClass
from portia.config_loader import ConfigLoader
from portia.model import GenerativeModel, LLMProvider
from portia.open_source_tools.llm_tool import LLMTool


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory and point loader defaults there."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / ".portia"
        config_dir.mkdir()
        with patch.object(ConfigLoader, "DEFAULT_CONFIG_DIR", config_dir):
            with patch.object(ConfigLoader, "DEFAULT_CONFIG_FILE", config_dir / "config.toml"):
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
def mock_portia_cls() -> MagicMock:
    """Mock the Portia class used by the CLI."""
    with patch("portia.cli.Portia", autospec=True) as mock_portia:
        yield mock_portia


@pytest.fixture(autouse=True)
def mock_config_get_generative_model() -> None:
    """Avoid real model initialization inside CLI."""
    
    with patch.object(Config, "get_generative_model") as mock_get_generative_model:
        mock_get_generative_model.return_value = MagicMock(spec=GenerativeModel)
        yield None


def test_cli_run_basic(mock_portia_cls: MagicMock, temp_config_dir: Path) -> None:
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
    mock_portia_cls: MagicMock,
    provider: str,
    expected_provider: LLMProvider,
    temp_config_dir: Path,
) -> None:
    """Test --llm-provider mapping to Config.llm_provider enum."""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["run", "Calculate 1 + 2", "--llm-provider", provider], input="y\n"
    )
    assert result.exit_code == 0
    assert mock_portia_cls.call_count == 1
    cfg = mock_portia_cls.call_args.kwargs["config"]
    assert cfg.llm_provider == expected_provider


def test_cli_run_config_set_planner_model(mock_portia_cls: MagicMock, temp_config_dir: Path) -> None:
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


def test_cli_run_config_multi_setting(mock_portia_cls: MagicMock, temp_config_dir: Path) -> None:
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


def test_cli_run_no_confirmation(mock_portia_cls: MagicMock, temp_config_dir: Path) -> None:
    """Test running without confirmation prompt."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "Compute 3 * 3", "--confirm", "false"])
    assert result.exit_code == 0

    mock_portia = mock_portia_cls.return_value
    assert mock_portia.plan.call_count == 1
    assert mock_portia.plan.call_args[0][0] == "Compute 3 * 3"
    assert mock_portia.run_plan.call_count == 1


def test_cli_run_custom_end_user_id(mock_portia_cls: MagicMock, temp_config_dir: Path) -> None:
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


def test_cli_run_reject_confirmation(mock_portia_cls: MagicMock, temp_config_dir: Path) -> None:
    """Test rejecting plan execution at the prompt."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "Subtract 5 - 3"], input="n\n")
    assert result.exit_code == 0

    mock_portia = mock_portia_cls.return_value
    assert mock_portia.plan.call_count == 1
    assert mock_portia.run_plan.call_count == 0


def test_cli_plan_default(mock_portia_cls: MagicMock, temp_config_dir: Path) -> None:
    """Test the CLI plan command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["plan", "What is the weather?"])
    assert result.exit_code == 0

    mock_portia = mock_portia_cls.return_value
    assert mock_portia.plan.call_count == 1
    assert mock_portia.plan.call_args[0][0] == "What is the weather?"


def test_cli_list_tools(temp_config_dir: Path) -> None:
    """Test listing available tools."""
    llm_tool = LLMTool()
    with patch("portia.cli.DefaultToolRegistry", autospec=True) as mock_tool_registry:
        mock_tool_registry.return_value.get_tools.return_value = [llm_tool]  
        runner = CliRunner()
        result = runner.invoke(cli, ["list-tools"])
    assert result.exit_code == 0
    assert llm_tool.pretty() in result.output


def test_cli_version(temp_config_dir: Path) -> None:
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
    with open(config_file, "w") as f:
        toml.dump(data, f)

    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("portia.cli._get_config") as mock_get_config:
            mock_cli_config = Mock()
            mock_config = Mock()

           
            def must_get_side_effect(*args, **kwargs):
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

            
            with patch("portia.tool_registry.DefaultToolRegistry") as MockRegistry:
                mock_registry = MockRegistry.return_value
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
                    
def test_cli_config_create_and_list(temp_config_dir):
        runner = CliRunner()
       
        result = runner.invoke(cli, ["config", "create", "myprofile"])
        assert result.exit_code == 0
        
        result = runner.invoke(cli, ["config", "list"])
        assert "myprofile" in result.output

def test_cli_config_set_and_get(temp_config_dir):
    runner = CliRunner()
    
    runner.invoke(cli, ["config", "create", "myprofile"])
    
    result = runner.invoke(cli, ["config", "set", "myprofile", "llm_provider=openai"])
    assert result.exit_code == 0
    
    result = runner.invoke(cli, ["config", "get", "myprofile", "llm_provider"])
    assert "openai" in result.output

def test_cli_config_set_default_and_get(temp_config_dir):
    runner = CliRunner()
    runner.invoke(cli, ["config", "create", "foo"])
    runner.invoke(cli, ["config", "create", "bar"])
   
    result = runner.invoke(cli, ["config", "set-default", "bar"])
    assert result.exit_code == 0
    
    result = runner.invoke(cli, ["config", "get", "bar"])
    assert result.exit_code == 0

def test_cli_config_delete(temp_config_dir):
    runner = CliRunner()
    runner.invoke(cli, ["config", "create", "todelete"])
   
    result = runner.invoke(cli, ["config", "delete", "todelete"], input="y\n")
    assert result.exit_code == 0
    result = runner.invoke(cli, ["config", "list"])
    assert "todelete" not in result.output

def test_cli_config_path(temp_config_dir):
    runner = CliRunner()
    result = runner.invoke(cli, ["config", "path"])
    assert result.exit_code == 0
    assert ".portia" in result.output

def test_cli_config_validate(temp_config_dir):
    runner = CliRunner()
    runner.invoke(cli, ["config", "create", "validprofile"])
    result = runner.invoke(cli, ["config", "validate", "validprofile"])
    assert result.exit_code == 0