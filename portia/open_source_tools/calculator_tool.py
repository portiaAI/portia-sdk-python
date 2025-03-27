"""Simple Calculator Implementation."""

import re
from tempfile import TemporaryDirectory
from typing import Any, Optional, Type

from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.file_management.copy import CopyFileTool, FileCopyInput
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from portia.errors import ToolHardError
from portia.tool import Tool, ToolRunContext

working_directory = TemporaryDirectory()
toolkit = FileManagementToolkit(
    root_dir=str(working_directory.name),
)  # If you don't provide a root_dir, operations will default to the current working directory
toolkit.get_tools()


class PortiaCopyFileTool(Tool[str]):
    """Copy a file to a new location."""

    id: str = "copy_file_tool"
    name: str = "Copy File Tool"
    description: str = "Copy a file to a new location."
    args_schema: type[BaseModel] = FileCopyInput
    output_schema: tuple[str, str] = ("str", "A string dump of the computed result")
    langchain_tool: CopyFileTool = Field(default_factory=CopyFileTool)

    def run(self, _: ToolRunContext, **other_args: str) -> str:
        """Run the tool."""
        return self.langchain_tool.run(other_args)


def to_portia(langchain_tool: BaseTool, tool_id: Optional[str] = None) -> Tool:
    name = getattr(langchain_tool, "name", "Unnamed Tool")
    description = getattr(langchain_tool, "description", "No description provided.")
    args_schema = getattr(langchain_tool, "args_schema", None)

    if tool_id is None:
        tool_id = name.lower().replace(" ", "_")

    class LangChainToolWrapper(Tool[str]):
        id: str = tool_id
        name: str = name
        description: str = description
        args_schema: Type[BaseModel] = args_schema
        output_schema: tuple[str, str] = ("str", "A string dump of the computed result")

        def run(self, _: ToolRunContext, **kwargs: Any) -> str:
            """Run the wrapped LangChain tool."""
            return langchain_tool.run(**kwargs)

    # Create an instance of the wrapper class
    return LangChainToolWrapper()

tool = to_portia(CopyFileTool)

class CalculatorToolSchema(BaseModel):
    """Input for the CalculatorTool."""

    math_question: str = Field(
        ...,
        description="The mathematical question to be evaluated in natural language",
    )


class CalculatorTool(Tool[float]):
    """Takes a basic maths question in natural language and returns the result.

    Works best for maths expressions containing only numbers and the operators +, -, *, x, /.
    """

    id: str = "calculator_tool"
    name: str = "Calculator Tool"
    description: str = "Takes a basic maths question and returns the result."
    "Works best for maths expressions containing only numbers and the operators +, -, *, x, /."
    args_schema: type[BaseModel] = CalculatorToolSchema
    output_schema: tuple[str, str] = ("str", "A string dump of the computed result")

    def run(self, _: ToolRunContext, math_question: str) -> float:
        """Run the CalculatorTool."""
        expression = self.math_expression(math_question)
        if not expression:
            raise ToolHardError("No valid mathematical expression found in the input.")

        try:
            result = eval(expression)  # noqa: S307
            return float(result)
        except ZeroDivisionError as e:
            raise ToolHardError("Error: Division by zero.") from e
        except Exception as e:
            raise ToolHardError(f"Error evaluating expression: {e}") from e

    def math_expression(self, prompt: str) -> str:  # noqa: C901, PLR0912
        """Convert words and phrases to standard operators."""
        prompt = prompt.lower()
        prompt = prompt.replace("added to", "+").replace("plus", "+").replace("and", "+")
        prompt = prompt.replace("minus", "-")
        prompt = prompt.replace("times", "*")
        prompt = prompt.replace("what is ", "").replace("?", "")
        prompt = prompt.replace("x", "*")  # Convert 'x' to '*' for multiplication

        # Handle "subtracted from" and "subtract from" separately
        if "subtracted from" in prompt:
            parts = prompt.split("subtracted from")
            if len(parts) == 2:  # noqa: PLR2004
                prompt = parts[1].strip() + " - " + parts[0].strip()
        elif "subtract" in prompt and "from" in prompt:
            match = re.search(r"subtract\s+(.+)\s+from\s+(.+)", prompt)
            if match:
                prompt = f"{match.group(2)} - {match.group(1)}"
        else:
            prompt = prompt.replace("subtract", "-")

        # Handle "divided by" and "divide by" separately
        if "divided by" in prompt:
            parts = prompt.split("divided by")
            if len(parts) == 2:  # noqa: PLR2004
                prompt = parts[0].strip() + " / " + parts[1].strip()
        elif "divide" in prompt and "by" in prompt:
            match = re.search(r"divide\s+(.+)\s+by\s+(.+)", prompt)
            if match:
                prompt = f"{match.group(1)} / {match.group(2)}"
        else:
            prompt = prompt.replace("divide", "/")

        # Handle "multiply by" and "multiplied by"
        if "multiply" in prompt and "by" in prompt:
            match = re.search(r"multiply\s+(.+)\s+by\s+(.+)", prompt)
            if match:
                prompt = f"{match.group(1)} * {match.group(2)}"
        elif "multiplied by" in prompt:
            parts = prompt.split("multiplied by")
            if len(parts) == 2:  # noqa: PLR2004
                prompt = parts[0].strip() + " * " + parts[1].strip()
        else:
            prompt = prompt.replace("multiply", "*")

        # Extract the mathematical expression
        return "".join(re.findall(r"[\d\+\-\*/\(\)\.\s]", prompt))
