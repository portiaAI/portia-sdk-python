"""Tool for responding to prompts and completing tasks that are related to image understanding."""

from __future__ import annotations

from typing import Any

from langchain.schema import HumanMessage
from pydantic import BaseModel, Field

from portia.llm_wrapper import LLMWrapper
from portia.tool import Tool, ToolRunContext


class ImageUnderstandingToolSchema(BaseModel):
    """Input for Image Understanding Tool."""

    task: str = Field(
        ...,
        description="The task to be completed by the Image tool.",
    )
    image_url: str = Field(
        ...,
        description="Image URL for processing.",
    )


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

    def run(self, ctx: ToolRunContext, **kwargs: Any) -> str:  # noqa: ANN401
        """Run the ImageTool."""
        image_wrapper = LLMWrapper(ctx.config)
        image_processor = image_wrapper.to_langchain()

        tool_schema = ImageUnderstandingToolSchema(**kwargs)

        # Define system and user messages
        context = (
            "Additional context for the Image tool to use to complete the task, provided by the "
            "plan run information and results of other tool calls. Use this to resolve any "
            "tasks"
        )
        if ctx.execution_context.plan_run_context:
            context += f"\nPlan run context: {ctx.execution_context.plan_run_context}"
        if self.tool_context:
            context += f"\nTool context: {self.tool_context}"
        content = (
            tool_schema.task
            if not len(context.split("\n")) > 1
            else f"{context}\n\n{tool_schema.task}"
        )

        messages = [
            HumanMessage(content=self.prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": content},
                    {
                        "type": "image_url",
                        "image_url": {"url": tool_schema.image_url},
                    },
                ],
            ),
        ]

        response = image_processor.invoke(messages)
        return str(response.content)
