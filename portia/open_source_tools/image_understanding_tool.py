"""Tool for responding to prompts and completing tasks that are related to image understanding."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Self

from langchain.schema import HumanMessage
from pydantic import BaseModel, Field, model_validator

from portia.errors import ToolHardError
from portia.model import GenerativeModel  # noqa: TC001 - used in Pydantic Schema
from portia.tool import Tool, ToolRunContext


class ImageUnderstandingToolSchema(BaseModel):
    """Input for Image Understanding Tool."""

    task: str = Field(
        ...,
        description="The task to be completed by the Image tool.",
    )
    image_url: str | None = Field(
        default=None,
        description="Image URL for processing.",
    )
    image_file: str | None = Field(
        default=None,
        description="Image file for processing.",
    )

    @model_validator(mode="after")
    def check_image_url_or_file(self) -> Self:
        """Check that only one of image_url or image_file is provided."""
        has_image_url = self.image_url is not None
        has_image_file = self.image_file is not None
        if not has_image_url ^ has_image_file:
            raise ValueError("One of image_url or image_file is required")
        return self


class ImageUnderstandingTool(Tool[str]):
    """General purpose image understanding tool. Customizable to user requirements."""

    id: str = "image_understanding_tool"
    name: str = "Image Understanding Tool"
    description: str = (
        "Tool for understanding images from a URL. Capable of tasks like object detection, "
        "OCR, scene recognition, and image-based Q&A. This tool uses its native capabilities "
        "to analyze images and provide insights."
    )
    args_schema: type[BaseModel] = ImageUnderstandingToolSchema
    output_schema: tuple[str, str] = (
        "str",
        "The Image understanding tool's response to the user query about the provided image.",
    )
    prompt: str = """
        You are an Image understanding tool used to analyze images and respond to queries.
        You can perform tasks like object detection, OCR, scene recognition, and image-based Q&A.
        Provide concise and accurate responses based on the image provided.
        """
    tool_context: str = ""

    model: GenerativeModel | None | str = Field(
        default=None,
        exclude=True,
        description="The model to use for the ImageUnderstandingTool. If not provided, "
        "the model will be resolved from the config.",
    )

    def run(self, ctx: ToolRunContext, **kwargs: Any) -> str:
        """Run the ImageTool."""
        model = ctx.config.get_generative_model(self.model) or ctx.config.get_default_model()

        tool_schema = ImageUnderstandingToolSchema(**kwargs)

        # Define system and user messages
        context = (
            "Additional context for the Image tool to use to complete the task, provided by the "
            "plan run information and results of other tool calls. Use this to resolve any "
            "tasks"
        )
        if self.tool_context:
            context += f"\nTool context: {self.tool_context}"
        content = (
            tool_schema.task
            if not len(context.split("\n")) > 1
            else f"{context}\n\n{tool_schema.task}"
        )

        if tool_schema.image_url:
            image_url = tool_schema.image_url
        elif tool_schema.image_file:  # pragma: no cover
            with Path(tool_schema.image_file).open("rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                mime_type = mimetypes.guess_type(tool_schema.image_file)[0]
                image_url = f"data:{mime_type};base64,{image_data}"
        else:  # pragma: no cover
            raise ToolHardError("No image URL or file provided")

        messages = [
            HumanMessage(content=self.prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": content},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            ),
        ]

        response = model.to_langchain().invoke(messages)
        return str(response.content)
