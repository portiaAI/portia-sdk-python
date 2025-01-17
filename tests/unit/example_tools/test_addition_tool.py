"""Addition tool tests."""

from portia.context import empty_context
from portia.example_tools.addition import AdditionTool


def test_addition_tool_successful_response() -> None:
    """Test that AdditionTool adds correctly."""
    tool = AdditionTool()
    ctx = empty_context()
    result = tool.run(ctx, 10, 13)
    assert result == 23
