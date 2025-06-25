"""Mock MCP server for testing."""

from __future__ import annotations

import sys
from logging import getLogger

from mcp.server import FastMCP

logger = getLogger(__name__)


server = FastMCP("server", port=11385, log_level="DEBUG")


@server.tool()
def add_one(input_number: float) -> str:
    """Add one to the input.

    Args:
        input_number: The input to add one to.

    Returns:
        The input plus one.

    """
    return str(input_number + 1)


if __name__ == "__main__":
    logger.info("Starting MCP server with args: %s", sys.argv)
    server.run(
        transport=sys.argv[1]  # type: ignore[arg-type]
        if len(sys.argv) > 1 and sys.argv[1] in ["stdio", "sse", "streamable-http"]
        else "stdio",
    )
