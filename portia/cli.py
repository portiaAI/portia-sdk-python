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
import sys
from enum import Enum
from functools import wraps
from types import UnionType
from typing import TYPE_CHECKING, Any, Callable, get_args

import click
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined
from typing_extensions import get_origin

from portia.clarification_handler import ClarificationHandler
from portia.config import Config, GenerativeModelsConfig
from portia.errors import InvalidConfigError
from portia.execution_context import execution_context
from portia.logger import logger
from portia.portia import ExecutionHooks, Portia
from portia.tool_registry import DefaultToolRegistry

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

    from portia.clarification import (
        ActionClarification,
        Clarification,
        CustomClarification,
        InputClarification,
        MultipleChoiceClarification,
        ValueConfirmationClarification,
    )

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

    end_user_id: str = Field(
        default="",
        description="The end user id to use in the execution context.",
    )

    confirm: bool = Field(
        default=True,
        description="Whether to confirm plans before running them.",
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

    field_type = _annotation_to_click_type(info.annotation)
    if field_type is None:
        return f

    optional_kwargs = {}
    default = info.default if info.default_factory is None else info.default_factory()  # type: ignore reportCallIssue
    if isinstance(default, Enum):
        optional_kwargs["default"] = default.name
    elif default is not None and default is not PydanticUndefined:
        optional_kwargs["default"] = default

    field_help = info.description or f"Set the value for {option_name}"

    return click.option(
        f"--{option_name}",
        type=field_type,
        help=field_help,
        **optional_kwargs,
    )(f)


def _annotation_to_click_type(  # noqa: PLR0911
    annotation: type[Any] | None,
) -> click.ParamType | None:
    """Convert a type annotation to a click type."""
    match annotation:
        case builtins.int:
            return click.INT
        case builtins.float:
            return click.FLOAT
        case builtins.bool:
            return click.BOOL
        case builtins.str:
            return click.STRING
        case builtins.list:
            return click.Tuple([str])
        case _ if isinstance(annotation, type) and issubclass(annotation, Enum):
            return click.Choice(
                [e.value for e in annotation],
                case_sensitive=False,
            )
        case _ if get_origin(annotation) is UnionType:
            args = get_args(annotation)
            for arg in args:
                if (click_type := _annotation_to_click_type(arg)) is not None:
                    return click_type
            return None
        case _:
            return None


def common_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """Define common options for CLI commands."""
    for field, info in Config.model_fields.items():
        generate_cli_option_from_pydantic_field(f, field, info)
    for field, info in GenerativeModelsConfig.model_fields.items():
        generate_cli_option_from_pydantic_field(f, field, info)
    for field, info in CLIConfig.model_fields.items():
        generate_cli_option_from_pydantic_field(f, field, info)

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return f(*args, **kwargs)

    return wrapper


class CLIExecutionHooks(ExecutionHooks):
    """Execution hooks for the CLI."""

    def __init__(self) -> None:
        """Set up execution hooks for the CLI."""
        super().__init__(clarification_handler=CLIClarificationHandler())


class CLIClarificationHandler(ClarificationHandler):
    """Handles clarifications by obtaining user input from the CLI."""

    def handle_action_clarification(
        self,
        clarification: ActionClarification,
        on_resolution: Callable[[Clarification, object], None],  # noqa: ARG002
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002
    ) -> None:
        """Handle an action clarification.

        Does this by showing the user the URL on the CLI and instructing them to click on
        it to proceed.
        """
        logger().info(
            click.style(
                f"{clarification.user_guidance} -- Please click on the link below to proceed."
                f"{clarification.action_url}",
                fg=87,
            ),
        )

    def handle_input_clarification(
        self,
        clarification: InputClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002
    ) -> None:
        """Handle a user input clarifications by asking the user for input from the CLI."""
        user_input = click.prompt(
            click.style(clarification.user_guidance + "\nPlease enter a value", fg=87),
        )
        return on_resolution(clarification, user_input)

    def handle_multiple_choice_clarification(
        self,
        clarification: MultipleChoiceClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002
    ) -> None:
        """Handle a multi-choice clarification by asking the user for input from the CLI."""
        choices = click.Choice(clarification.options)
        user_input = click.prompt(
            click.style(clarification.user_guidance + "\nPlease choose a value:\n", fg=87),
            type=choices,
        )
        return on_resolution(clarification, user_input)

    def handle_value_confirmation_clarification(
        self,
        clarification: ValueConfirmationClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a value confirmation clarification by asking the user to confirm from the CLI."""
        if click.confirm(text=click.style(clarification.user_guidance, fg=87), default=False):
            on_resolution(clarification, True)  # noqa: FBT003
        else:
            on_error(clarification, "Clarification was rejected by the user")

    def handle_custom_clarification(
        self,
        clarification: CustomClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002
    ) -> None:
        """Handle a custom clarification."""
        click.echo(click.style(clarification.user_guidance, fg=87))
        click.echo(click.style(f"Additional data: {json.dumps(clarification.data)}", fg=87))
        user_input = click.prompt(click.style("\nPlease enter a value:\n", fg=87))
        return on_resolution(clarification, user_input)


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
def run(
    query: str,
    **kwargs,  # noqa: ANN003
) -> None:
    """Run a query."""
    cli_config, config = _get_config(**kwargs)

    # Add the tool registry
    registry = DefaultToolRegistry(config)

    # Run the query
    portia = Portia(
        config=config,
        tools=(
            registry.match_tools(tool_ids=[cli_config.tool_id]) if cli_config.tool_id else registry
        ),
        execution_hooks=CLIExecutionHooks(),
    )

    with execution_context(end_user_id=cli_config.end_user_id):
        plan = portia.plan(query)

        if cli_config.confirm:
            click.echo(plan.pretty_print())
            if not click.confirm("Do you want to execute the plan?"):
                return

        plan_run = portia.run_plan(plan)
        click.echo(plan_run.model_dump_json(indent=4))


@click.command()
@common_options
@click.argument("query")
def plan(
    query: str,
    **kwargs,  # noqa: ANN003
) -> None:
    """Plan a query."""
    cli_config, config = _get_config(**kwargs)
    portia = Portia(config=config)

    with execution_context(end_user_id=cli_config.end_user_id):
        output = portia.plan(query)

    click.echo(output.model_dump_json(indent=4))


@click.command()
@common_options
def list_tools(
    **kwargs,  # noqa: ANN003
) -> None:
    """List tools."""
    cli_config, config = _get_config(**kwargs)

    for tool in DefaultToolRegistry(config).get_tools():
        click.echo(tool.model_dump_json(indent=4))


def _get_config(
    **kwargs,  # noqa: ANN003
) -> tuple[CLIConfig, Config]:
    """Init config."""
    cli_config = CLIConfig(**kwargs)
    if cli_config.env_location == EnvLocation.ENV_FILE:
        load_dotenv(override=True)
    try:
        config = Config.from_default(**kwargs)
    except InvalidConfigError as e:
        logger().error(e.message)
        sys.exit(1)

    return (cli_config, config)


cli.add_command(version)
cli.add_command(run)
cli.add_command(plan)
cli.add_command(list_tools)

if __name__ == "__main__":
    cli(obj={})
