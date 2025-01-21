"""Tools module.

This module defines an abstract base class for tools that can be extended to create custom tools
Each tool has a unique ID and a name, and child classes should implement the `run` method
with their specific logic.
"""

from __future__ import annotations

import json
import time
from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, Self

import httpx
from langchain_core.tools import StructuredTool
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    SecretStr,
    ValidationError,
    field_serializer,
    model_validator,
)

from portia.agents.base_agent import Output
from portia.clarification import (
    ActionClarification,
    Clarification,
    InputClarification,
    MultiChoiceClarification,
)
from portia.common import SERIALIZABLE_TYPE_VAR
from portia.errors import InvalidToolDescriptionError, ToolHardError, ToolSoftError
from portia.logger import logger
from portia.templates.render import render_template

if TYPE_CHECKING:
    from portia.context import ExecutionContext

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
    def run(
        self,
        ctx: ExecutionContext,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> SERIALIZABLE_TYPE_VAR | Clarification:
        """Run the tool.

        This method must be implemented by subclasses to define the tool's specific behavior.

        Args:
            ctx (ExecutionContext): Context of the execution environment
            args (Any): The arguments passed to the tool for execution.
            kwargs (Any): The keyword arguments passed to the tool for execution.

        Returns:
            Any: The result of the tool's execution.

        """

    def _run(
        self,
        ctx: ExecutionContext,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> Output[SERIALIZABLE_TYPE_VAR] | Output[list[Clarification]]:
        """Run the Tool function and generate an Output object with descriptions."""
        args_dict = {f"{i}": arg for i, arg in enumerate(args)}
        data = {**args_dict, **kwargs}
        logger().info(f"Invoking: {self.name} with {data} and context {ctx}")
        start_time = time.time()

        try:
            output = self.run(ctx, *args, **kwargs)
        except Exception as e:
            logger().error(f"Tool: {self.name} returned error {e}")
            # check if error is wrapped as a Hard or Soft Tool Error.
            # if not wrap as ToolSoftError
            if not isinstance(e, ToolHardError) and not isinstance(e, ToolSoftError):
                raise ToolSoftError(e) from e
            raise
        else:
            execution_time = time.time() - start_time
            logger().debug(f"Tool {self.name} executed in {execution_time:.2f} seconds")
            logger().info("Tool output: {output}", output=output)

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
        ctx: ExecutionContext,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[str, Output[SERIALIZABLE_TYPE_VAR]]:
        """Run the Tool function and generate an Output object with descriptions.

        Returns a tuple of the output and an Output object, as expected by langchain tools.
        This allows us to capture the output (artifact) directly instead of having it
        serialized to a string first (see content_and_artifact in langgraph tool definition).
        """
        intermediate_output = self._run(ctx, *args, **kwargs)
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
    def check_description_length(self) -> Self:
        """Check that the description is less than 1024 characters."""
        # OpenAI has a max function description length of 1024 characters.
        description_length = len(self._generate_tool_description())
        if description_length > MAX_TOOL_DESCRIPTION_LENGTH:
            raise InvalidToolDescriptionError(self.name)
        return self

    def to_langchain(self, ctx: ExecutionContext, return_artifact: bool = False) -> StructuredTool:  # noqa: FBT001, FBT002
        """Return a LangChain representation of this tool.

        Langchain agent needs to use the "content" response format, but Langgraph
        prefers the other.
        """
        if return_artifact:
            return StructuredTool(
                name=self.name.replace(" ", "_"),
                description=self._generate_tool_description(),
                args_schema=self.args_schema,
                func=partial(self._run_with_artifacts, ctx),
                return_direct=True,
                response_format="content_and_artifact",
            )
        return StructuredTool(
            name=self.name.replace(" ", "_"),
            description=self._generate_tool_description(),
            args_schema=self.args_schema,
            func=partial(self._run, ctx),
        )

    def args_json_schema(self) -> dict[str, Any]:
        """Return the json_schema for the tool args."""
        return self.args_schema.model_json_schema()

    def __str__(self) -> str:
        """Return the string representation."""
        return (
            f"ToolModel(id={self.id!r}, name={self.name!r}, "
            f"description={self.description!r}, "
            f"args_schema={self.args_schema.__name__!r}, "
            f"output_schema={self.output_schema!r})"
        )

    @field_serializer("args_schema")
    def serialize_args_schema(self, value: type[BaseModel]) -> str:
        """Serialize the type by returning its class name."""
        return value.__name__


class PortiaRemoteTool(Tool, Generic[SERIALIZABLE_TYPE_VAR]):
    """Tool that passes run execution to Portia Cloud."""

    api_key: SecretStr
    api_endpoint: str

    def parse_response(self, response: dict[str, Any]) -> Output:
        """Parse a JSON response into domain models/errors."""
        output = Output.model_validate(response["output"])
        # Handle Tool Errors
        if isinstance(output.value, str):
            if "ToolSoftError" in output.value:
                raise ToolSoftError(output.value)
            if "ToolHardError" in output.value:
                raise ToolHardError(output.value)
        # Handle Clarifications
        if isinstance(output.value, list) and output.value and "type" in output.value[0]:
            clarification = output.value[0]
            match clarification["type"]:
                case "Action Clarification":
                    return Output(
                        value=ActionClarification(
                            action_url=HttpUrl(clarification["action_url"]),
                            user_guidance=clarification["user_guidance"],
                        ),
                    )
                case "Input Clarification":
                    return Output(
                        value=InputClarification(
                            argument_name=clarification["argument_name"],
                            user_guidance=clarification["user_guidance"],
                        ),
                    )
                case "Multi Choice Clarification":
                    return Output(
                        value=MultiChoiceClarification(
                            argument_name=clarification["argument_name"],
                            user_guidance=clarification["user_guidance"],
                            options=clarification["options"],
                        ),
                    )
        return output

    def run(
        self,
        ctx: ExecutionContext,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  #  noqa: ANN401
    ) -> SERIALIZABLE_TYPE_VAR | None | Clarification:
        """Invoke the run endpoint and handle response."""
        try:
            # Combine args and kwargs
            args_dict = {f"{i}": arg for i, arg in enumerate(args)}
            data = {**args_dict, **kwargs}
            # Send to Cloud
            response = httpx.post(
                url=f"{self.api_endpoint}/api/v0/tools/{self.id}/run/",
                content=json.dumps(
                    {
                        "arguments": data,
                        "execution_context": {
                            "end_user_id": ctx.end_user_id or "",
                            "additional_data": ctx.additional_data or {},
                        },
                    },
                ),
                headers={
                    "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                timeout=60,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger().error(f"Error from Portia Cloud: {e.response.content}")
            raise ToolHardError(e) from e
        except Exception as e:
            logger().error(f"Unhandled error from Portia Cloud: {e}")
            raise ToolHardError(e) from e
        else:
            try:
                output = self.parse_response(response.json())
            except (ValidationError, KeyError) as e:
                logger().error(f"Error parsing response from Portia Cloud: {e}")
                raise ToolHardError(e) from e
            else:
                return output.value
