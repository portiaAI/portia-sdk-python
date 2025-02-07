"""Simple Example."""

import json

from pydantic import BaseModel, Field

from portia.config import Config, LogLevel
from portia.open_source_tools.llm_tool import LLMTool
from portia.runner import Runner
from portia.tool import Tool, ToolRunContext
from portia.tool_registry import InMemoryToolRegistry


class PythonExecutionToolSchema(BaseModel):
    """Input for ClarificationTool."""

    code_snippet: str = Field(description="The python code to execute.")


class PythonExecutionTool(Tool):
    """Executes python code."""

    id: str = "python_execution_tool"
    name: str = "Python Execution Tool"
    description: str = (
        "Executes well formed Python Code. Input to this tool must be well formed python."
        " With all variables defined within the code snippet."
        " See the python_code_creator tool for creation."
        " Please make sure the output of the script is assigned into a variable called output."
    )
    args_schema: type[BaseModel] = PythonExecutionToolSchema
    output_schema: tuple[str, str] = (
        "str",
        "str: The return value of the code",
    )

    def run(
        self,
        ctx: ToolRunContext,  # noqa: ARG002
        code_snippet: str,
    ) -> str:
        """Run the code."""
        context = {}
        exec(code_snippet, locals=context)  # noqa: S102
        output = context.get("output")
        return json.dumps(output)


additional_tools = InMemoryToolRegistry.from_local_tools(
    [
        PythonExecutionTool(),
        LLMTool(
            id="python_code_creator",
            name="Python code creator",
            description="Used to generate python code",
            prompt="You are an expert python coder."
            " Your job is to write code that solves the problem given to you in the prompt."
            " Assign the final solution to the problem into a variable called output."
            " Only return the code. The returned value should be directly executable."
            " Don't include ` or any other formatting.",
        ),
    ],
)

runner = Runner(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tools=additional_tools,
)


# Simple Example
workflow = runner.execute_query("What is the area of a triangle with height 12.3cm and base 6.43cm")
print(workflow.outputs)
