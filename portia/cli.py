"""CLI Implementation.

Usage:

portia-cli run "add 4 + 8" - run a query
portia-cli plan "add 4 + 8" - plan a query
"""

import click

from portia.config import default_config
from portia.example_tools import example_tool_registry
from portia.runner import Runner


@click.group()
def cli() -> None:
    """Portia CLI."""


@click.command()
@click.argument("query")
def run(query: str) -> None:
    """Run a query."""
    config = default_config()
    runner = Runner(config=config, tool_registry=example_tool_registry)
    output = runner.run_query(query, tools=[])
    click.echo(output)


@click.command()
@click.argument("query")
def plan(query: str) -> None:
    """Plan a query."""
    config = default_config()
    runner = Runner(config=config, tool_registry=example_tool_registry)
    output = runner.plan_query(query, tools=[])
    click.echo(output)


cli.add_command(run)
cli.add_command(plan)

if __name__ == "__main__":
    cli()
