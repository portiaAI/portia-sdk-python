"""CLI Implementation.

Usage:

portia-cli run "add 4 + 8" - run a query
portia-cli plan "add 4 + 8" - plan a query
"""

from __future__ import annotations

import os
import webbrowser
from enum import Enum
from functools import wraps
from typing import Any, Callable

import click
from dotenv import load_dotenv

from portia.clarification import ActionClarification, InputClarification, MultiChoiceClarification
from portia.config import Config, LLMModel, LLMProvider, LogLevel
from portia.context import execution_context
from portia.logger import logger
from portia.open_source_tools import example_tool_registry
from portia.runner import Runner
from portia.tool_registry import PortiaToolRegistry
from portia.workflow import WorkflowState


class EnvLocation(Enum):
    """The location of the environment variables."""

    ENV_FILE = "ENV_FILE"
    ENV_VARS = "ENV_VARS"


class CLIOptions(Enum):
    """The options for the CLI."""

    LOG_LEVEL = "LOG_LEVEL"
    LLM_PROVIDER = "LLM_PROVIDER"
    LLM_MODEL = "LLM_MODEL"
    ENV_LOCATION = "ENV_LOCATION"


PORTIA_API_KEY = "portia_api_key"


def common_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """Define common options for CLI commands."""

    @click.option(
        "--log-level",
        type=click.Choice([level.name for level in LogLevel], case_sensitive=False),
        default=LogLevel.INFO.value,
        help="Set the logging level",
    )
    @click.option(
        "--llm-provider",
        type=click.Choice([p.value for p in LLMProvider], case_sensitive=False),
        required=False,
        help="The LLM provider to use",
    )
    @click.option(
        "--env-location",
        type=click.Choice([e.value for e in EnvLocation], case_sensitive=False),
        default=EnvLocation.ENV_VARS.value,
        help="The location of the environment variables: default is environment variables",
    )
    @click.option(
        "--llm-model",
        type=click.Choice([m.value for m in LLMModel], case_sensitive=False),
        required=False,
        help="The LLM model to use",
    )
    @click.option(
        "--end-user-id",
        type=click.STRING,
        required=False,
        help="Run with an end user id",
    )
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return f(*args, **kwargs)

    return wrapper


@click.group()
def cli() -> None:
    """Portia CLI."""


@click.command()
@common_options
@click.argument("query")
def run(  # noqa: PLR0913
    query: str,
    log_level: str,
    llm_provider: str | None,
    llm_model: str | None,
    end_user_id: str | None,
    env_location: str,
) -> None:
    """Run a query."""
    config = _get_config(
        log_level=LogLevel[log_level.upper()],
        llm_provider=LLMProvider(llm_provider.upper()) if llm_provider else None,
        llm_model=LLMModel(llm_model.upper()) if llm_model else None,
        env_location=EnvLocation(env_location.upper()),
    )
    # Add the tool registry
    registry = example_tool_registry
    if config.has_api_key(PORTIA_API_KEY):
        registry += PortiaToolRegistry(config)

    # Run the query
    runner = Runner(config=config, tool_registry=registry)

    with execution_context(end_user_id=end_user_id):
        workflow = runner.execute_query(query)

        final_states = [WorkflowState.COMPLETE, WorkflowState.FAILED]
        while workflow.state not in final_states:
            for clarification in workflow.get_outstanding_clarifications():
                if isinstance(clarification, MultiChoiceClarification):
                    user_input = input(
                        clarification.user_guidance
                        + "\nPlease enter an option from below:\n"
                        + "\n".join(clarification.options)
                        + "\nchoice: ",
                    )
                    clarification.resolve(user_input)
                if isinstance(clarification, ActionClarification):
                    webbrowser.open(str(clarification.action_url))
                    logger().info("Please complete authentication to continue")
                    auth_complete = False
                    while not auth_complete:
                        user_input = input("Is Authentication Complete [Y/N]")
                        if user_input.lower() == "y":
                            auth_complete = True
                    clarification.resolve(None)
                if isinstance(clarification, InputClarification):
                    user_input = input(
                        clarification.user_guidance + "\nPlease enter a value:\n",
                    )
                    clarification.resolve(user_input)
            runner.execute_workflow(workflow)

        click.echo(workflow.model_dump_json(indent=4))


@click.command()
@common_options
@click.argument("query")
def plan(
    query: str,
    log_level: str,
    llm_provider: str | None,
    llm_model: str | None,
    env_location: str,
) -> None:
    """Plan a query."""
    config = _get_config(
        log_level=LogLevel[log_level.upper()],
        llm_provider=LLMProvider(llm_provider.upper()) if llm_provider else None,
        llm_model=LLMModel(llm_model.upper()) if llm_model else None,
        env_location=EnvLocation(env_location.upper()),
    )
    registry = example_tool_registry
    if config.has_api_key(PORTIA_API_KEY):
        registry += PortiaToolRegistry(config)
    runner = Runner(config=config, tool_registry=registry)
    output = runner.generate_plan(query)
    click.echo(output.model_dump_json(indent=4))


def _get_config(
    log_level: LogLevel,
    llm_provider: LLMProvider | None,
    llm_model: LLMModel | None,
    env_location: EnvLocation,
) -> Config:
    """Get the config from the context."""
    if env_location == EnvLocation.ENV_FILE:
        load_dotenv(override=True)

    keys = [
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("MISTRAL_API_KEY"),
    ]

    keys = [k for k in keys if k is not None]
    if len(keys) > 1 and llm_provider is None and llm_model is None:
        message = "Multiple LLM keys found, but no default provided: Select a provider or model"
        raise click.UsageError(message)

    if llm_provider or llm_model:
        provider = (
            llm_provider
            if llm_provider
            # we are sure that llm_model is not None at this point
            else llm_model.provider()  # pyright: ignore[reportOptionalMemberAccess]
        )
        model = llm_model if llm_model else provider.default_model()
        config = Config.from_default(
            llm_provider=provider,
            llm_model_name=model,
            default_log_level=log_level,
        )
    else:
        config = Config.from_default(default_log_level=log_level)

    return config


cli.add_command(run)
cli.add_command(plan)

if __name__ == "__main__":
    cli(obj={})  # Pass empty dict as the initial context object
