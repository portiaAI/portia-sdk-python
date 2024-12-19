"""CLI."""

import click

from portia.config import Config
from portia.runner import Runner
from portia.tool_registry import InMemoryToolRegistry
from tests.utils import AdditionTool


@click.group()
def cli() -> None:
    """Portia CLI."""


@click.command()
@click.argument("query")
def run(query: str) -> None:
    """Run a query."""
    config = Config()
    tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool()])
    runner = Runner(config=config, tool_registry=tool_registry)
    output = runner.run_query(query, tools=[])
    click.echo(output)


@click.command()
@click.argument("query")
def plan(query: str) -> None:
    """Plan a query."""
    config = Config()
    tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool()])
    runner = Runner(config=config, tool_registry=tool_registry)
    output = runner.plan_query(query, tools=[])
    click.echo(output)


cli.add_command(run)
cli.add_command(plan)

if __name__ == "__main__":
    cli()
