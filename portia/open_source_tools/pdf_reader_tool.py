"""Tool for reading PDF files and extracting text content using Mistral OCR."""

import os
from pathlib import Path

from mistralai import Mistral
from pydantic import BaseModel, Field

from portia.tool import Tool, ToolRunContext


class PDFReaderToolSchema(BaseModel):
    """Input for PDFReaderTool."""

    file_path: str = Field(
        ...,
        description=("The path to the PDF file to be read."),
    )


class PDFReaderTool(Tool[str]):
    """Read a PDF file and extract its text content using Mistral OCR."""

    id: str = "pdf_reader_tool"
    name: str = "PDF Reader Tool"
    description: str = "Read a PDF file and extract its text content using Mistral OCR"
    args_schema: type[BaseModel] = PDFReaderToolSchema
    output_schema: tuple[str, str] = (
        "str",
        "The extracted text content from the PDF file.",
    )

    def run(self, _: ToolRunContext, file_path: str) -> str:
        """Run the PDFReaderTool."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found at path: {file_path}")

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")

        client = Mistral(api_key=api_key)
        with Path(file_path).open("rb") as pdf_file:
            uploaded_pdf = client.files.upload(
                file={
                    "file_name": f"{file_path}",
                    "content": pdf_file,
                },
                purpose="ocr",
            )
        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
        )

        return "\n".join(page.markdown for page in ocr_response.pages)
