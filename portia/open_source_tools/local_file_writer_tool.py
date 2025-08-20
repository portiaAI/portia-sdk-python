"""Local file writer tool."""

from pathlib import Path

from pydantic import BaseModel, Field

from portia.tool import Tool, ToolRunContext


class FileWriterToolSchema(BaseModel):
    """Schema defining the inputs for the FileWriterTool."""

    filename: str = Field(
        ...,
        description="The location where the file should be saved",
    )
    content: str = Field(
        ...,
        description="The content to write to the file",
    )


class FileWriterTool(Tool[str]):
    """Writes content to a file."""

    id: str = "file_writer_tool"
    name: str = "File writer tool"
    description: str = "Writes content to a file locally"
    args_schema: type[BaseModel] = FileWriterToolSchema
    output_schema: tuple[str, str] = ("str", "A string indicating where the content was written to")

    def run(self, _: ToolRunContext, filename: str, content: str) -> str:
        """Run the FileWriterTool."""
        filepath = Path(filename)
        if filepath.is_file():
            with Path.open(filepath, "w", encoding="utf-8") as file:
                file.write(content)
        else:
            with Path.open(filepath, "x", encoding="utf-8") as file:
                file.write(content)
        return f"Content written to {filename}"
