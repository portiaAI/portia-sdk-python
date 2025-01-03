"""CLI Implementation.

Usage:

portia-cli run "add 4 + 8" - run a query
portia-cli plan "add 4 + 8" - plan a query
"""

import click
from pydantic import BaseModel, Field

from portia.config import default_config
from portia.runner import Runner
from portia.tool import Tool
from portia.tool_registry import InMemoryToolRegistry


class AdditionToolSchema(BaseModel):
    """Input for AdditionTool."""

    a: int = Field(..., description="The first number to add")
    b: int = Field(..., description="The second number to add")


class AdditionTool(Tool):
    """Adds two numbers."""

    id: str = "add_tool"
    name: str = "Add Tool"
    description: str = "Takes two numbers and adds them together"
    args_schema: type[BaseModel] = AdditionToolSchema
    output_schema: tuple[str, str] = ("int", "int: The value of the addition")

    def run(self, a: int, b: int) -> int:
        """Add the numbers."""
        return a + b


@click.group()
def cli() -> None:
    """Portia CLI."""


@click.command()
@click.argument("query")
def run(query: str) -> None:
    """Run a query."""
    config = default_config()
    tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool()])
    runner = Runner(config=config, tool_registry=tool_registry)
    output = runner.run_query(query, tools=[])
    click.echo(output)


@click.command()
@click.argument("query")
def plan(query: str) -> None:
    """Plan a query."""
    config = default_config()
    tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool()])
    runner = Runner(config=config, tool_registry=tool_registry)
    output = runner.plan_query(query, tools=[])
    click.echo(output)


cli.add_command(run)
cli.add_command(plan)

if __name__ == "__main__":
    cli()
