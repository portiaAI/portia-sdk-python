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
@click.option(
    "--llm-model",
    type=click.Choice([m.value for m in LLMModel], case_sensitive=False),
    required=False,
    help="The LLM model to use",
)
def run(
    query: str,
    llm_provider: LLMProvider | None,
    llm_model: LLMModel | None,
    env_location: EnvLocation,
) -> None:
    """Run a query."""
    env_location = EnvLocation(env_location)
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
        provider = LLMProvider(llm_provider) if llm_provider else LLMModel(llm_model).provider()
        model = LLMModel(llm_model) if llm_model else provider.default_model()
        config = Config.from_default(
            llm_provider=provider,
            llm_model_name=model,
            default_log_level=LogLevel.ERROR,
        )
    else:
        config = Config.from_default(default_log_level=LogLevel.ERROR)

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
