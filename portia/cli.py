"""CLI Implementation.

Usage:

portia-cli run "add 4 + 8" - run a query
portia-cli plan "add 4 + 8" - plan a query
"""

from __future__ import annotations

from enum import Enum

import click
from dotenv import load_dotenv

from portia.config import Config, LLMProvider, LogLevel
from portia.example_tools import example_tool_registry
from portia.runner import Runner
from portia.tool_registry import PortiaToolRegistry


class EnvLocation(Enum):
    """The location of the environment variables."""

    ENV_FILE = "env_file"
    ENV_VARS = "env_vars"


@click.group()
def cli() -> None:
    """Portia CLI."""


@click.command()
@click.argument("query")
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
def run(query: str, llm_provider: LLMProvider | None, env_location: EnvLocation) -> None:
    """Run a query."""
    env_location = EnvLocation(env_location)
    if env_location == EnvLocation.ENV_FILE:
        load_dotenv(override=True)
    config = Config.from_default(default_log_level=LogLevel.ERROR)

    # Check if we have multiple LLM keys and no LLM provider is provided
    keys = [config.openai_api_key, config.anthropic_api_key, config.mistralai_api_key]
    keys = [k for k in keys if k is not None]
    if len(keys) > 1 and llm_provider is None:
        raise click.UsageError("Must provide a LLM provider when using multiple LLM keys")

    # Add the tool registry
    registry = example_tool_registry
    if config.has_api_key("portia_api_key"):
        registry += PortiaToolRegistry(config)

    # Run the query
    runner = Runner(config=config, tool_registry=registry)
    output = runner.run_query(query)
    click.echo(output.model_dump_json(indent=4))


@click.command()
@click.argument("query")
def plan(query: str) -> None:
    """Plan a query."""
    config = Config.from_default(default_log_level=LogLevel.ERROR)
    registry = example_tool_registry
    if config.has_api_key("portia_api_key"):
        registry += PortiaToolRegistry(config)
    runner = Runner(config=config, tool_registry=registry)
    output = runner.plan_query(query)
    click.echo(output.model_dump_json(indent=4))


cli.add_command(run)
cli.add_command(plan)

if __name__ == "__main__":
    cli()
