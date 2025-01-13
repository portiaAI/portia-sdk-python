"""CLI Implementation.

Usage:

portia-cli run "add 4 + 8" - run a query
portia-cli plan "add 4 + 8" - plan a query
"""

from __future__ import annotations

import os
from enum import Enum

import click
from dotenv import load_dotenv

from portia.config import Config, LLMModel, LLMProvider, LogLevel
from portia.example_tools import example_tool_registry
from portia.runner import Runner
from portia.tool_registry import PortiaToolRegistry


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

@click.group()
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
@click.pass_context
def cli(ctx: click.Context,
        log_level: str,
        llm_provider: str | None,
        llm_model: str | None,
        env_location: str) -> None:
    """Portia CLI."""

    ctx.ensure_object(dict)
    ctx.obj[CLIOptions.LOG_LEVEL.name] = LogLevel[log_level.upper()]
    ctx.obj[CLIOptions.LLM_PROVIDER.name] = LLMProvider(llm_provider.upper()) if llm_provider else None
    ctx.obj[CLIOptions.LLM_MODEL.name] = LLMModel(llm_model.upper()) if llm_model else None
    ctx.obj[CLIOptions.ENV_LOCATION.name] = EnvLocation(env_location.upper())


@click.command()
@click.argument("query")
@click.pass_context
def run(ctx: click.Context, query: str) -> None:
    """Run a query."""
    config = _get_config(ctx)
    # Add the tool registry
    registry = example_tool_registry
    if config.has_api_key(PORTIA_API_KEY):
        registry += PortiaToolRegistry(config)

    # Run the query
    runner = Runner(config=config, tool_registry=registry)
    output = runner.run_query(query)
    click.echo(output.model_dump_json(indent=4))

@click.command()
@click.argument("query")
@click.pass_context
def plan(ctx: click.Context, query: str) -> None:
    """Plan a query."""
    config = _get_config(ctx)
    registry = example_tool_registry
    if config.has_api_key(PORTIA_API_KEY):
        registry += PortiaToolRegistry(config)
    runner = Runner(config=config, tool_registry=registry)
    output = runner.plan_query(query)
    click.echo(output.model_dump_json(indent=4))

def _get_config(ctx: click.Context) -> Config:
    """Get the config from the context."""

    log_level = ctx.obj.get(CLIOptions.LOG_LEVEL.name, LogLevel.INFO)
    llm_provider = ctx.obj.get(CLIOptions.LLM_PROVIDER.name, None)
    llm_model = ctx.obj.get(CLIOptions.LLM_MODEL.name, None)
    env_location = ctx.obj.get(CLIOptions.ENV_LOCATION.name, EnvLocation.ENV_VARS)
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
        provider = llm_provider if llm_provider else llm_model.provider()
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
