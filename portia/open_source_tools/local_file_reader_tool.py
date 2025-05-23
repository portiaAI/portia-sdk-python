"""Tool for reading files from disk."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from portia.clarification import Clarification, MultipleChoiceClarification
from portia.errors import ToolHardError
from portia.tool import Tool, ToolRunContext


class FileReaderToolSchema(BaseModel):
    """Schema defining the inputs for the FileReaderTool."""

    filename: str = Field(
        ...,
        description="The path (either full or relative) where the file should be read from",
    )


class FileReaderTool(Tool[str]):
    """Finds and reads content from a local file on Disk."""

    id: str = "file_reader_tool"
    name: str = "File reader tool"
    description: str = "Finds and reads content from a local file on Disk"
    args_schema: type[BaseModel] = FileReaderToolSchema
    output_schema: tuple[str, str] = ("str", "A string dump or JSON of the file content")

    def run(self, ctx: ToolRunContext, filename: str) -> str | Clarification:  # noqa: PLR0911
        """Run the FileReaderTool."""
        file_path = Path(filename)
        suffix = file_path.suffix.lower()

        if file_path.is_file():
            match suffix:
                case ".csv":
                    return pd.read_csv(file_path).to_string()
                case ".json":
                    with file_path.open("r", encoding="utf-8") as json_file:
                        return json.dumps(json.load(json_file), indent=4)
                case ".xls":
                    return pd.read_excel(file_path).to_string()
                case ".xlsx":
                    return pd.read_excel(file_path).to_string()
                case ".txt":
                    return file_path.read_text(encoding="utf-8")
                case ".log":
                    return file_path.read_text(encoding="utf-8")
                case _:
                    raise ToolHardError(
                        f"Unsupported file format: {suffix}."
                        "Supported formats are .txt, .log, .csv, .json, .xls, .xlsx.",
                    )

        alt_file_paths = self.find_file(file_path)
        if alt_file_paths:
            return MultipleChoiceClarification(
                plan_run_id=ctx.plan_run.id,
                argument_name="filename",
                user_guidance=(
                    f"Found {filename} in these location(s). "
                    f"Pick one to continue:\n{alt_file_paths}"
                ),
                options=alt_file_paths,
                source="File reader tool",
            )

        raise ToolHardError(f"No file found on disk with the path {filename}.")

    def find_file(self, file_path: Path) -> list[str]:
        """Return a full file path or None."""
        search_path = file_path.parent
        filename = file_path.name
        return [str(filepath) for filepath in search_path.rglob(filename) if filepath.is_file()]
