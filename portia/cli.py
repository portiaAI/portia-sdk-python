"""CLI Implementation.

Usage:

portia-cli run "add 4 + 8" - run a query
portia-cli plan "add 4 + 8" - plan a query
portia-cli list-tools
"""

from __future__ import annotations

import builtins
import importlib.metadata
import json
import os
import webbrowser
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import click
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from portia.clarification import (
    ActionClarification,
    CustomClarification,
    InputClarification,
    MultipleChoiceClarification,
    ValueConfirmationClarification,
)
from portia.config import Config, StorageClass
from portia.execution_context import execution_context
from portia.logger import logger
from portia.open_source_tools.registry import get_default_tool_registry
from portia.runner import Runner
from portia.workflow import WorkflowState

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

DEFAULT_FILE_PATH = ".portia"
PORTIA_API_KEY = "portia_api_key"


class EnvLocation(Enum):
    """The location of the environment variables."""

    ENV_FILE = "ENV_FILE"
    ENV_VARS = "ENV_VARS"


class CLIConfig(BaseModel):
    """Config for the CLI."""

    env_location: EnvLocation = Field(
        default=EnvLocation.ENV_VARS,
        description="The location of the environment variables.",
    )

    config_file: str = Field(
        default=f"{DEFAULT_FILE_PATH}/config.json",
        description="The location of the JSON config file for the CLI to use.",
    )

    end_user_id: str = Field(
        default="",
        description="The end user id to use in the execution context.",
    )

    confirm: bool = Field(
        default=True,
        description="Whether to confirm plans before running them.",
    )

    output_path: str = Field(
        default=DEFAULT_FILE_PATH,
        description="Where to output to",
    )

    tool_id: str | None = Field(
        default=None,
        description="The tool ID to use. If not provided, all tools will be used.",
    )


def generate_cli_option_from_pydantic_field(
    f: Callable[..., Any],
    field: str,
    info: FieldInfo,
) -> Callable[..., Any]:
    """Generate a click option from a pydantic field."""
    option_name = field.replace("_", "-")

    # Don't support passing API Keys as options as it leaks them to history/logs etc
    if option_name.endswith("api-key"):
        return f

    field_type = click.STRING
    field_default = info.default
    if info.default_factory:
        field_default = info.default_factory()  # type: ignore  # noqa: PGH003

    match info.annotation:
        case builtins.int:
            field_type = click.INT
        case builtins.float:
            field_type = click.FLOAT
        case builtins.bool:
            field_type = click.BOOL
        case builtins.str:
            field_type = click.STRING
        case builtins.list:
            field_type = click.Tuple([str])
        case _:
            if isinstance(info.annotation, type) and issubclass(info.annotation, Enum):
                field_type = click.Choice(
                    [e.name for e in info.annotation],
                    case_sensitive=False,
                )
                field_default = info.default.name if info.default else None

    field_help = info.description or f"Set the value for {option_name}"

    return click.option(
        f"--{option_name}",
        type=field_type,
        default=field_default,
        help=field_help,
    )(f)


def common_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """Define common options for CLI commands."""
    for field, info in Config.model_fields.items():
        generate_cli_option_from_pydantic_field(f, field, info)
    for field, info in CLIConfig.model_fields.items():
        generate_cli_option_from_pydantic_field(f, field, info)

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return f(*args, **kwargs)

    return wrapper


@click.group(context_settings={"max_content_width": 240})
def cli() -> None:
    """Portia CLI."""


@click.command()
def version() -> None:
    """Print the CLI tool version."""
    click.echo(importlib.metadata.version("portia-sdk-python"))


@click.command()
@common_options
@click.argument("query")
def run(  # noqa: C901
    query: str,
    **kwargs,  # noqa: ANN003
) -> None:
    """Run a query."""
    cli_config, config = _get_config(**kwargs)

    # Add the tool registry
    registry = get_default_tool_registry(config)

    # Run the query
    runner = Runner(
        config=config,
        tools=(
            registry.match_tools(tool_ids=[cli_config.tool_id]) if cli_config.tool_id else registry
        ),
    )

    with execution_context(end_user_id=cli_config.end_user_id):
        plan = runner.generate_plan(query)

        if cli_config.confirm:
            click.echo(plan.model_dump_json(indent=4))
            if not click.confirm("Do you want to execute the plan?"):
                return

        workflow = runner.create_workflow(plan)
        workflow = runner.execute_workflow(workflow)

        final_states = [WorkflowState.COMPLETE, WorkflowState.FAILED]
        while workflow.state not in final_states:
            for clarification in workflow.get_outstanding_clarifications():
                if isinstance(clarification, MultipleChoiceClarification):
                    choices = click.Choice(clarification.options)
                    user_input = click.prompt(
                        clarification.user_guidance + "\nPlease choose a value:\n",
                        type=choices,
                    )
                    workflow = runner.resolve_clarification(clarification, user_input, workflow)
                if isinstance(clarification, ActionClarification):
                    webbrowser.open(str(clarification.action_url))
                    logger().info("Please complete authentication to continue")
                    workflow = runner.wait_for_ready(workflow)
                if isinstance(clarification, InputClarification):
                    user_input = click.prompt(
                        clarification.user_guidance + "\nPlease enter a value:\n",
                    )
                    workflow = runner.resolve_clarification(clarification, user_input, workflow)
                if isinstance(clarification, ValueConfirmationClarification):
                    if click.confirm(text=clarification.user_guidance, default=False):
                        workflow = runner.resolve_clarification(
                            clarification,
                            response=True,
                            workflow=workflow,
                        )
                    else:
                        workflow.state = WorkflowState.FAILED
                        runner.storage.save_workflow(workflow)

                if isinstance(clarification, CustomClarification):
                    click.echo(clarification.user_guidance)
                    click.echo(f"Additional data: {json.dumps(clarification.data)}")
                    user_input = click.prompt("\nPlease enter a value:\n")
                    workflow = runner.resolve_clarification(clarification, user_input, workflow)

            runner.execute_workflow(workflow)

        click.echo(workflow.model_dump_json(indent=4))


@click.command()
@common_options
@click.argument("query")
def plan(
    query: str,
    **kwargs,  # noqa: ANN003
) -> None:
    """Plan a query."""
    cli_config, config = _get_config(**kwargs)
    runner = Runner(config=config)

    with execution_context(end_user_id=cli_config.end_user_id):
        output = runner.generate_plan(query)

    click.echo(output.model_dump_json(indent=4))


@click.command()
@common_options
def list_tools(
    **kwargs,  # noqa: ANN003
) -> None:
    """List tools."""
    cli_config, config = _get_config(**kwargs)

    for tool in get_default_tool_registry(config):
        click.echo(tool.model_dump_json(indent=4))


@click.command()
@common_options
def config_write(
    **kwargs,  # noqa: ANN003
) -> None:
    """Write config file to disk."""
    cli_config, config = _get_config(**kwargs)

    output_path = Path(cli_config.output_path, "config.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_contents = config.model_dump_json(indent=4)

    with output_path.open("w") as f:
        f.write(file_contents)


def _get_config(
    **kwargs,  # noqa: ANN003
) -> tuple[CLIConfig, Config]:
    """Init config."""
    cli_config = CLIConfig(**kwargs)
    if cli_config.env_location == EnvLocation.ENV_FILE:
        load_dotenv(override=True)
    config = Config.from_default(**kwargs)

    keys = [
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("MISTRAL_API_KEY"),
    ]

    llm_provider = config.llm_provider
    llm_model = config.llm_model_name

    keys = [k for k in keys if k is not None]
    if len(keys) > 1 and llm_provider is None and llm_model is None:
        message = "Multiple LLM keys found, but no default provided: Select a provider or model"
        raise click.UsageError(message)

    if os.getenv("PORTIA_API_KEY"):
        config.storage_class = StorageClass.CLOUD
        config.portia_api_endpoint = os.getenv("PORTIA_API_ENDPOINT") or config.portia_api_endpoint

    if llm_provider or llm_model:
        config.llm_provider = llm_provider if llm_provider else llm_model.provider()
        config.llm_model_name = (
            llm_model
            if llm_model in llm_provider.associated_models()
            else config.llm_provider.default_model()
        )

    return (cli_config, config)


cli.add_command(version)
cli.add_command(run)
cli.add_command(plan)
cli.add_command(list_tools)
cli.add_command(config_write)

if __name__ == "__main__":
    cli(obj={})
