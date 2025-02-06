"""FileWriterTool tests."""

from pathlib import Path

import pytest

from portia.open_source_tools.local_file_writer_tool import FileWriterTool
from tests.utils import get_test_tool_context


def test_file_writer_tool_successful_write(tmp_path: Path) -> None:
    """Test that FileWriterTool successfully writes content to a file."""
    tool = FileWriterTool()
    ctx = get_test_tool_context()
    filename = tmp_path / "test_file.txt"
    content = "Hello, world!"

    result = tool.run(ctx, str(filename), content)
    assert filename.read_text() == content
    assert result == f"Content written to {filename}"


def test_file_writer_tool_overwrite_existing_file(tmp_path: Path) -> None:
    """Test that FileWriterTool overwrites an existing file."""
    tool = FileWriterTool()
    ctx = get_test_tool_context()
    filename = tmp_path / "existing_file.txt"
    filename.write_text("Old content")
    content = "New content!"

    result = tool.run(ctx, str(filename), content)
    assert filename.read_text() == content
    assert result == f"Content written to {filename}"


def test_file_writer_tool_handles_file_creation_error(tmp_path: Path) -> None:
    """Test that FileWriterTool raises an error if file creation fails."""
    tool = FileWriterTool()
    ctx = get_test_tool_context()
    filename = tmp_path / "error_file.txt"

    # Make the directory read-only to simulate permission error
    tmp_path.chmod(0o400)
    with pytest.raises(OSError, match="Permission denied"):
        tool.run(ctx, str(filename), "This should fail")
