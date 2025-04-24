"""Tests for PDF reader tool."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from portia.open_source_tools.pdf_reader_tool import PDFReaderTool
from tests.utils import get_test_tool_context


@pytest.fixture
def pdf_reader_tool() -> PDFReaderTool:
    """Fixture to create an instance of PDFReaderTool."""
    return PDFReaderTool()


@pytest.fixture
def mock_mistral_client() -> MagicMock:
    """Fixture to create a mock for the Mistral client."""
    mock = MagicMock()
    # Setup the OCR response structure
    pages = [MagicMock(markdown="Page 1 content"), MagicMock(markdown="Page 2 content")]
    mock.ocr.process.return_value = MagicMock(pages=pages)
    # Setup the signed URL
    mock.files.get_signed_url.return_value = MagicMock(url="https://signed-url.com")
    # Setup the upload response
    mock.files.upload.return_value = MagicMock(id="file_id_123")
    return mock


@patch("portia.open_source_tools.pdf_reader_tool.Mistral")
@patch.dict(os.environ, {"MISTRAL_API_KEY": "test_api_key"})
def test_pdf_reader_tool_run_success(
    mock_mistral_class: MagicMock,
    mock_mistral_client: MagicMock,
    pdf_reader_tool: PDFReaderTool,
    tmp_path: Path,
) -> None:
    """Test that PDFReaderTool runs successfully and returns text content."""
    mock_mistral_class.return_value = mock_mistral_client

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"dummy pdf content")

    result = pdf_reader_tool.run(get_test_tool_context(), str(pdf_path))

    mock_mistral_class.assert_called_once_with(api_key="test_api_key")
    mock_mistral_client.files.upload.assert_called_once()
    assert mock_mistral_client.files.upload.call_args[1]["purpose"] == "ocr"
    mock_mistral_client.ocr.process.assert_called_once()
    assert mock_mistral_client.ocr.process.call_args[1]["model"] == "mistral-ocr-latest"
    assert result == "Page 1 content\nPage 2 content"


def test_pdf_reader_tool_file_not_found(pdf_reader_tool: PDFReaderTool) -> None:
    """Test that PDFReaderTool raises an error when file is not found."""
    with pytest.raises(FileNotFoundError):
        pdf_reader_tool.run(get_test_tool_context(), "/non/existent/file.pdf")


@patch.dict(os.environ, {})  # Empty environment to ensure MISTRAL_API_KEY is not set
def test_pdf_reader_tool_missing_api_key(
    pdf_reader_tool: PDFReaderTool,
    tmp_path: Path,
) -> None:
    """Test that PDFReaderTool raises an error when MISTRAL_API_KEY is not set."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"dummy pdf content")

    with pytest.raises(ValueError, match="MISTRAL_API_KEY environment variable is not set"):
        pdf_reader_tool.run(get_test_tool_context(), str(pdf_path))
