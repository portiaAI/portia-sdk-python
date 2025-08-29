"""FileReaderTool tests."""

import json
from pathlib import Path

import pandas as pd
import pytest

from portia.clarification import MultipleChoiceClarification, UserVerificationClarification
from portia.errors import ToolHardError
from portia.open_source_tools.local_file_reader_tool import FileReaderTool
from tests.utils import get_test_tool_context


def test_file_reader_tool_read_txt(tmp_path: Path) -> None:
    """Test that FileReaderTool reads content from a .txt file."""
    tool = FileReaderTool()
    ctx = get_test_tool_context()
    filename = tmp_path / "test.txt"
    content = "Hello, world!"
    filename.write_text(content, encoding="utf-8")

    result = tool.run(ctx, str(filename))
    assert result == content


def test_file_reader_tool_read_log(tmp_path: Path) -> None:
    """Test that FileReaderTool reads content from a .log file."""
    tool = FileReaderTool()
    ctx = get_test_tool_context()
    filename = tmp_path / "test.log"
    content = "Hello, world!"
    filename.write_text(content, encoding="utf-8")

    result = tool.run(ctx, str(filename))
    assert result == content


def test_file_reader_tool_read_json(tmp_path: Path) -> None:
    """Test that FileReaderTool reads content from a .json file."""
    tool = FileReaderTool()
    ctx = get_test_tool_context()
    filename = tmp_path / "test.json"
    content = {"key": "value"}
    filename.write_text(json.dumps(content), encoding="utf-8")

    result = tool.run(ctx, str(filename))
    assert isinstance(result, str)
    assert json.loads(result) == content


def test_file_reader_tool_read_csv(tmp_path: Path) -> None:
    """Test that FileReaderTool reads content from a .csv file."""
    tool = FileReaderTool()
    ctx = get_test_tool_context()
    filename = tmp_path / "test.csv"
    frame = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    frame.to_csv(filename, index=False)

    result = tool.run(ctx, str(filename))
    assert isinstance(result, str)
    assert "col1" in result
    assert "col2" in result


def test_file_reader_tool_read_xlsx(tmp_path: Path) -> None:
    """Test that FileReaderTool reads content from a .xlsx file."""
    tool = FileReaderTool()
    ctx = get_test_tool_context()
    filename = tmp_path / "test.xlsx"
    frame = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    frame.to_excel(filename, index=False)

    result = tool.run(ctx, str(filename))
    assert isinstance(result, str)
    assert "col1" in result
    assert "col2" in result


def test_file_reader_tool_read_xls(tmp_path: Path) -> None:
    """Test that FileReaderTool reads content from a .xls file."""
    tool = FileReaderTool()
    ctx = get_test_tool_context()
    filename = tmp_path / "test.xls"
    frame = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    frame.to_excel(filename, index=False)

    result = tool.run(ctx, str(filename))
    assert isinstance(result, str)
    assert "col1" in result
    assert "col2" in result


def test_file_reader_tool_unsupported_format(tmp_path: Path) -> None:
    """Test that FileReaderTool raises an error for unsupported file formats."""
    tool = FileReaderTool()
    ctx = get_test_tool_context()
    filename = tmp_path / "test.unsupported"
    filename.write_text("Some content", encoding="utf-8")

    with pytest.raises(ToolHardError, match="Unsupported file format"):
        tool.run(ctx, str(filename))


def test_file_reader_tool_file_alt_files(tmp_path: Path) -> None:
    """Test that FileReaderTool returns UserVerificationClarification for single file match."""
    tool = FileReaderTool()
    ctx = get_test_tool_context()
    filename = tmp_path / "non_existent.txt"

    subfolder = tmp_path / "test"
    subfolder.mkdir()

    alt_filename = subfolder / "non_existent.txt"
    content = "Hello, world!"
    alt_filename.write_text(content, encoding="utf-8")

    output = tool.run(ctx, str(filename))
    # Single file match should return UserVerificationClarification
    assert isinstance(output, UserVerificationClarification)
    assert "Do you want to read this file?" in output.user_guidance
    assert str(alt_filename) in output.user_guidance


def test_file_reader_tool_multiple_alt_files(tmp_path: Path) -> None:
    """Test that FileReaderTool returns MultipleChoiceClarification for multiple file matches."""
    tool = FileReaderTool()
    ctx = get_test_tool_context()
    filename = tmp_path / "non_existent.txt"

    # Create multiple matching files
    subfolder1 = tmp_path / "test1"
    subfolder1.mkdir()
    alt_filename1 = subfolder1 / "non_existent.txt"
    alt_filename1.write_text("Hello from file 1!", encoding="utf-8")

    subfolder2 = tmp_path / "test2"
    subfolder2.mkdir()
    alt_filename2 = subfolder2 / "non_existent.txt"
    alt_filename2.write_text("Hello from file 2!", encoding="utf-8")

    output = tool.run(ctx, str(filename))
    # Multiple file matches should return MultipleChoiceClarification
    assert isinstance(output, MultipleChoiceClarification)
    assert isinstance(output.options, list)
    assert len(output.options) == 2
    assert str(alt_filename1) in output.options
    assert str(alt_filename2) in output.options
    assert "Pick one to continue:" in output.user_guidance
    assert str(filename) in output.user_guidance


def test_file_reader_tool_file_no_files(tmp_path: Path) -> None:
    """Test that FileReaderTool raises an error when file is not found."""
    tool = FileReaderTool()
    ctx = get_test_tool_context()
    filename = tmp_path / "non_existent.txt"

    with pytest.raises(ToolHardError):
        tool.run(ctx, str(filename))
