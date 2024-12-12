"""Tests for the Tool class."""

import pytest

from portia.tool import Tool


class MockTool(Tool[int]):
    """A mock tool subclass for testing."""

    def run(self, string: str) -> int:
        """Override the run method for testing."""
        return len(string)


@pytest.fixture
def mock_tool() -> MockTool:
    """Fixture to create a mock tool instance."""
    return MockTool(name="Test Tool")


def test_tool_initialization(mock_tool: MockTool) -> None:
    """Test initialization of a Tool."""
    assert mock_tool.name == "Test Tool"
    assert len(mock_tool.id) == 36  # noqa: PLR2004 - UUID length


def test_run_method(mock_tool: MockTool) -> None:
    """Test the run method of the MockTool."""
    test_string = "test string"
    result = mock_tool.run(test_string)
    assert result == len(test_string)

    result = mock_tool.run("")
    assert result == 0
