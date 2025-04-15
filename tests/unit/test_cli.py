"""Integration tests for the CLI."""

import re
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from portia.cli import cli
from portia.config import StorageClass
from portia.model import LLMProvider
from portia.open_source_tools.llm_tool import LLMTool


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture
def mock_portia_cls() -> Iterator[MagicMock]:
    """Mock the Portia class."""
    with patch("portia.cli.Portia", autospec=True) as mock_portia:
        yield mock_portia


@pytest.fixture
def mock_execution_context() -> Iterator[MagicMock]:
    """Mock the execution context."""
    with patch("portia.cli.execution_context", autospec=True) as mock:
        yield mock


def test_cli_run(mock_portia_cls: MagicMock) -> None:
    """Test the CLI --run command."""
    mock_portia = mock_portia_cls.return_value
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "Calculate 1 + 2"], input="y\n")
    assert result.exit_code == 0

    assert mock_portia.plan.call_count == 1
    assert mock_portia.plan.call_args[0][0] == "Calculate 1 + 2"
    assert mock_portia.run_plan.call_count == 1
    assert mock_portia.run_plan.call_args[0][0] is mock_portia.plan.return_value


def test_cli_run_config_set_provider(mock_portia_cls: MagicMock) -> None:
    """Test the CLI --set-provider command."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "Calculate 1 + 2", "--llm-provider", "anthropic"],
        input="y\n",
    )
    assert result.exit_code == 0
    assert mock_portia_cls.call_count == 1
    config = mock_portia_cls.call_args.kwargs["config"]
    assert config.llm_provider == LLMProvider.ANTHROPIC


def test_cli_run_config_set_planner_model(mock_portia_cls: MagicMock) -> None:
    """Test the CLI --planning-model argument."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "Calculate 1 + 2", "--planning-model", "openai/gpt-3.5-turbo"],
        input="y\n",
    )
    assert result.exit_code == 0
    assert mock_portia_cls.call_count == 1
    config = mock_portia_cls.call_args.kwargs["config"]
    assert config.models.planning_model == "openai/gpt-3.5-turbo"


@pytest.mark.xfail(reason="TODO: tool-id setting is not working, broken enum")
def test_cli_run_config_multi_setting(mock_portia_cls: MagicMock) -> None:
    """Test the CLI --planning-model argument."""
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
    config = mock_portia_cls.call_args.kwargs["config"]
    assert config.models.planning_model == "openai/gpt-3.5-turbo"
    assert config.llm_provider == LLMProvider.ANTHROPIC
    assert config.storage_class == StorageClass.MEMORY
    assert config.tool_id == "llm_tool"


def test_cli_run_no_confirmation(mock_portia_cls: MagicMock) -> None:
    """Test the CLI run command with confirmation disabled.

    This test invokes the CLI run command with --confirm set to false so that the confirmation
    prompt is skipped, and then ensures that both the planning and execution steps are called.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "Compute 3 * 3", "--confirm", "false"], input="")
    assert result.exit_code == 0

    mock_portia = mock_portia_cls.return_value
    assert mock_portia.plan.call_count == 1
    assert mock_portia.plan.call_args[0][0] == "Compute 3 * 3"
    assert mock_portia.run_plan.call_count == 1


def test_cli_run_custom_end_user_id(
    mock_portia_cls: MagicMock,
    mock_execution_context: MagicMock,
) -> None:
    """Test the CLI run command with a custom end user id.

    This test passes a custom end user id to the CLI and ensures that it is stored in the config
    and used when creating the Portia instance.
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "Sum 1 + 1", "--end-user-id", "user-123", "--confirm", "false"],
        input="",
    )
    assert result.exit_code == 0

    assert mock_execution_context.call_args_list[0].kwargs["end_user_id"] == "user-123"

    mock_portia = mock_portia_cls.return_value
    assert mock_portia.plan.call_args[0][0] == "Sum 1 + 1"
    assert mock_portia.run_plan.call_count == 1


def test_cli_run_reject_confirmation(mock_portia_cls: MagicMock) -> None:
    """Test the CLI run command when the user rejects plan execution.

    This test simulates the user entering 'n' at the confirmation prompt, which should
    result in the execution step (run_plan) not being called.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "Subtract 5 - 3"], input="n\n")
    assert result.exit_code == 0

    mock_portia = mock_portia_cls.return_value
    assert mock_portia.plan.call_count == 1
    assert mock_portia.run_plan.call_count == 0


def test_cli_plan_default(mock_portia_cls: MagicMock) -> None:
    """Test the CLI plan command with default configuration.

    This test invokes the plan command without extra config options,
    checking that the plan method is called with the query.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["plan", "What is the weather?"], input="")
    assert result.exit_code == 0

    mock_portia = mock_portia_cls.return_value
    assert mock_portia.plan.call_count == 1
    assert mock_portia.plan.call_args[0][0] == "What is the weather?"


def test_cli_list_tools() -> None:
    """Test the CLI list-tools command."""
    llm_tool = LLMTool()
    with patch("portia.cli.DefaultToolRegistry", autospec=True) as mock_tool_registry:
        mock_tool_registry.return_value.get_tools.return_value = [
            llm_tool,
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["list-tools"], input="")
    assert result.exit_code == 0
    assert llm_tool.name in result.output


def test_cli_version() -> None:
    """Test the CLI version command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["version"], input="")
    assert result.exit_code == 0
    assert re.match(r"\d+\.\d+\.\d+-?\w*", result.output) is not None
