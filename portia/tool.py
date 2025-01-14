"""Tools module.

This module defines an abstract base class for tools that can be extended to create custom tools
Each tool has a unique ID and a name, and child classes should implement the `run` method
with their specific logic.
"""

from __future__ import annotations

import json
import time
from abc import abstractmethod
from typing import Any, Generic

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, SecretStr, model_validator

from portia.agents.base_agent import Output
from portia.clarification import Clarification
from portia.common import SERIALIZABLE_TYPE_VAR
from portia.errors import InvalidToolDescriptionError, ToolHardError, ToolSoftError
from portia.logger import logger
from portia.templates.render import render_template

MAX_TOOL_DESCRIPTION_LENGTH = 1024


class _ArgsSchemaPlaceholder(BaseModel):
    pass


class Tool(BaseModel, Generic[SERIALIZABLE_TYPE_VAR]):
    """Abstract base class for a tool.

    This class serves as the blueprint for all tools. Child classes must implement the `run` method.

    Attributes:
        id (str): A unique identifier for the tool.
        name (str): The name of the tool.
        description (str): Purpose of the tool and usage.

    """

    id: str = Field(description="ID of the tool")
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Purpose of the tool and usage")
    args_schema: type[BaseModel] = Field(default_factory=lambda _: _ArgsSchemaPlaceholder)
    output_schema: tuple[str, str] = Field(
        ...,
        description="Output schema of the tool",
        examples=["(TYPE, DESCRIPTION)", "(json, json with API response, single object)"],
    )

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> SERIALIZABLE_TYPE_VAR | Clarification:  # noqa: ANN401
        """Run the tool.

        This method must be implemented by subclasses to define the tool's specific behavior.

        Args:
            args (Any): The arguments passed to the tool for execution.
            kwargs (Any): The keyword arguments passed to the tool for execution.

        Returns:
            Any: The result of the tool's execution.

        """

    def _run(
        self,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> Output[SERIALIZABLE_TYPE_VAR] | Output[list[Clarification]]:
        """Run the Tool function and generate an Output object with descriptions."""
        args_dict = {f"{i}": arg for i, arg in enumerate(args)}
        data = {**args_dict, **kwargs}
        logger.info(f"Invoking: {self.name} with {data}")
        start_time = time.time()
        try:
            output = self.run(*args, **kwargs)
        except Exception as e:
            # check if error is wrapped as a Hard or Soft Tool Error.
            # if not wrap as ToolSoftError
            if not isinstance(e, ToolHardError) and not isinstance(e, ToolSoftError):
                raise ToolSoftError(e) from e
            raise
        else:
            execution_time = time.time() - start_time
            logger.debug(f"Tool {self.name} executed in {execution_time:.2f} seconds")
            logger.info("Tool output: {output}", output=output)

        # handle clarifications cleanly
        if isinstance(output, Clarification) or (
            isinstance(output, list)
            and len(output) > 0
            and all(isinstance(item, Clarification) for item in output)
        ):
            clarifications = output if isinstance(output, list) else [output]
            return Output[list[Clarification]](
                value=clarifications,
            )
        return Output[SERIALIZABLE_TYPE_VAR](value=output)  # type: ignore  # noqa: PGH003

    def _run_with_artifacts(
        self,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[str, Output[SERIALIZABLE_TYPE_VAR]]:
        """Run the Tool function and generate an Output object with descriptions.

        Returns a tuple of the output and an Output object, as expected by langchain tools.
        This allows us to capture the output (artifact) directly instead of having it
        serialized to a string first (see content_and_artifact in langgraph tool definition).
        """
        intermediate_output = self._run(*args, **kwargs)
        return (intermediate_output.value, intermediate_output)  # type: ignore  # noqa: PGH003

    def _generate_tool_description(self) -> str:
        """Generate tool descriptions."""
        args = []
        args_description = []
        args_name_description_dict = []
        out_type = self.output_schema[0]
        out_description = self.output_schema[1]
        schema = self.args_json_schema()
        for arg, attribute in schema["properties"].items():
            arg_dict = {
                "name": arg,
                "type": attribute.get("type", None),
                "description": attribute.get("description", None),
                "required": arg in schema.get("required", []),
            }
            args_name_description_dict.append(arg_dict)
            if "type" in attribute:
                args.append(f"{arg}: '{attribute['type']}'")
            if "description" in attribute:
                args_description.append(f"{arg}: '{attribute['description']}")

        description = self.description.replace("\n", " ")
        overview = f"{self.name.replace(' ', '_')}({', '.join(args)})"

        if out_type:
            overview += f" -> {out_type}"

        template_dict = {
            "overview": overview,
            "overview_description": description,
            "args": args_name_description_dict,
            "output_description": out_description,
        }

        return render_template(
            "tool_description.xml.jinja",
            tool=template_dict,
        )

    @model_validator(mode="after")
    def check_description_length(self) -> Tool:
        """Check that the description is less than 1024 characters."""
        # OpenAI has a max function description length of 1024 characters.
        description_length = len(self._generate_tool_description())
        if description_length > MAX_TOOL_DESCRIPTION_LENGTH:
            raise InvalidToolDescriptionError(self.name)
        return self

    def to_langchain(self, return_artifact: bool = False) -> StructuredTool:  # noqa: FBT001, FBT002
        """Return a LangChain representation of this tool.

        Langchain agent needs to use the "content" response format, but Langgraph
        prefers the other.
        """
        if return_artifact:
            return StructuredTool(
                name=self.name.replace(" ", "_"),
                description=self._generate_tool_description(),
                args_schema=self.args_schema,
                func=self._run_with_artifacts,
                return_direct=True,
                response_format="content_and_artifact",
            )
        return StructuredTool(
            name=self.name.replace(" ", "_"),
            description=self._generate_tool_description(),
            args_schema=self.args_schema,
            func=self._run,
        )

    def args_json_schema(self) -> dict[str, Any]:
        """Return the json_schema for the tool args."""
        return self.args_schema.model_json_schema()


class PortiaRemoteTool(Tool, Generic[SERIALIZABLE_TYPE_VAR]):
    """Tool that passes run execution to Portia Cloud."""

    api_key: SecretStr
    api_endpoint: str

    def run(self, *args: Any, **kwargs: Any) -> SERIALIZABLE_TYPE_VAR | Clarification:  # noqa: ANN401
        """Invoke the run endpoint and handle response."""
        try:
            # Combine args and kwargs
            args_dict = {f"{i}": arg for i, arg in enumerate(args)}
            data = {**args_dict, **kwargs}
            # Send to Cloud
            response = httpx.post(
                url=f"{self.api_endpoint}/api/v0/tools/{self.id}/run/",
                content=json.dumps({"arguments": data, "execution_context": {}}),
                headers={
                    "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                timeout=60,
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Error from Portia Cloud: {e}")
            raise ToolHardError(e) from e
        else:
            return response.json()
