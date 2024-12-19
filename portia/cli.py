"""CLI."""

import click

from portia.config import Config
from portia.runner import Runner
from portia.tool_registry import InMemoryToolRegistry
from tests.utils import AdditionTool


@click.command()
@click.argument("query")
def cli(query: str) -> None:
    """Run a query."""
    config = Config()
    tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool()])
    runner = Runner(config=config, tool_registry=tool_registry)
    output = runner.run_query(query, tools=[])
    click.echo(output)


if __name__ == "__main__":
    cli()
