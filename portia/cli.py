"""CLI Implementation.

Usage:

portia-cli run "add 4 + 8" - run a query
portia-cli plan "add 4 + 8" - plan a query
"""

import click

from portia.config import Config, LogLevel
from portia.example_tools import example_tool_registry
from portia.runner import Runner
from portia.tool_registry import PortiaToolRegistry

LOG_LEVEL = "log_level"
PORTIA_API_KEY = "portia_api_key"

@click.group()
@click.option(
    "--log-level",
    type=click.Choice([level.name for level in LogLevel], case_sensitive=False),
    default="INFO",
    help="Set the logging level",
)
@click.pass_context
def cli(ctx: click.Context, log_level: str) -> None:
    """Portia CLI."""
    # Store log level in context for use in subcommands.
    ctx.ensure_object(dict)
    # Convert string to LogLevel enum
    ctx.obj[LOG_LEVEL] = LogLevel[log_level.upper()]


@click.command()
@click.argument("query")
@click.pass_context
def run(ctx: click.Context, query: str) -> None:
    """Run a query."""
    log_level = ctx.obj.get(LOG_LEVEL, LogLevel.INFO)
    config = Config.from_default(default_log_level=log_level)
    registry = example_tool_registry
    if config.has_api_key(PORTIA_API_KEY):
        registry += PortiaToolRegistry(config)
    runner = Runner(config=config, tool_registry=registry)
    output = runner.run_query(query)
    click.echo(output.model_dump_json(indent=4))


@click.command()
@click.argument("query")
@click.pass_context
def plan(ctx: click.Context, query: str) -> None:
    """Plan a query."""
    log_level = ctx.obj.get(LOG_LEVEL, LogLevel.INFO)
    config = Config.from_default(default_log_level=log_level)
    registry = example_tool_registry
    if config.has_api_key(PORTIA_API_KEY):
        registry += PortiaToolRegistry(config)
    runner = Runner(config=config, tool_registry=registry)
    output = runner.plan_query(query)
    click.echo(output.model_dump_json(indent=4))


cli.add_command(run)
cli.add_command(plan)

if __name__ == "__main__":
    cli(obj={})  # Pass empty dict as the initial context object
