"""Tools module.

This module defines an abstract base class for tools, providing a structure for creating custom
tools that can integrate with external systems. It includes an implementation of a base `Tool` class
that defines common attributes and behaviors, such as a unique ID and name. Child classes should
implement the `run` method to define the specific logic for interacting with the external systems
or performing actions.

The module also contains `PortiaRemoteTool`, a subclass of `Tool`, which implements the logic to
interact with Portia Cloud, including handling API responses and tool errors.

The tools in this module are designed to be extendable, allowing users to create their own tools
while relying on common functionality provided by the base class.
"""

from __future__ import annotations

import json
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
    MultipleChoiceClarification,
    ValueConfirmationClarification,
)
from portia.common import SERIALIZABLE_TYPE_VAR, PortiaBaseModel, combine_args_kwargs
from portia.context import ExecutionContext
from portia.errors import InvalidToolDescriptionError, ToolHardError, ToolSoftError
from portia.logger import logger
from portia.templates.render import render_template

if TYPE_CHECKING:
    from portia.context import ExecutionContext

"""MAX_TOOL_DESCRIPTION_LENGTH is the max length tool descriptions can be to respect API limits."""
MAX_TOOL_DESCRIPTION_LENGTH = 1024


class _ArgsSchemaPlaceholder(BaseModel):
    """Placeholder ArgsSchema for tools that take no arguments."""


class Tool(PortiaBaseModel, Generic[SERIALIZABLE_TYPE_VAR]):
    """Abstract base class for a tool.

    This class serves as the blueprint for all tools. Child classes must implement the `run` method.

    Attributes
    ----------
    id : str
        A unique identifier for the tool.
        This must be unique as collisions in a tool registry will lead to errors.
    name : str
        The name of the tool. The name is informational only but useful for debugging.
    description : str
        Purpose of the tool and usage.
        This is important information for the planner module to know when and how to use this tool.
    args_schema : type[BaseModel]
        The schema defining the expected input arguments for the tool.
        We use Pydantic models to define these types.
    output_schema : tuple[str, str]
        A tuple containing the type and description of the tool's output.
        To maximize the advantages of using an agentic approach this doesn't need to be
        tightly defined. Instead it should give just a high level overview of the type and
        contents of the tools output.
    should_summarize : bool
        Indicates whether the tool's output should be automatically summarized by the
        summarizer agent. For some tools summarization is useful (for example: a tool
        that fetches the latest news) whereas other tools it's not (for example: a tool
        that fetches raw price data).

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
    should_summarize: bool = Field(
        default=False,
        description="Whether the tool's output requires a summary. "
        "Tools may not require a summary if they already produce a nice textual output.",
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
            Any: The result of the tool's execution which can be any serializable type
            or a clarification.

        """

    def _run(
        self,
        ctx: ExecutionContext,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> Output[SERIALIZABLE_TYPE_VAR] | Output[list[Clarification]]:
        """Invoke the Tool.run function and handle converting the result into an Output object.

        This is the entry point for agents to invoke a tool.

        Args:
            ctx (ExecutionContext): The execution context for the tool.
            *args (Any): Additional positional arguments for the tool function.
            **kwargs (Any): Additional keyword arguments for the tool function.

        Returns:
            Output[SERIALIZABLE_TYPE_VAR] | Output[list[Clarification]]: The tool's output wrapped
            in an Output object.

        Raises:
            ToolSoftError: If an error occurs and it is not already a Hard or Soft Tool error.

        """
        try:
            output = self.run(ctx, *args, **kwargs)
        except Exception as e:
            # check if error is wrapped as a Hard or Soft Tool Error.
            # if not wrap as ToolSoftError
            if not isinstance(e, ToolHardError) and not isinstance(e, ToolSoftError):
                raise ToolSoftError(e) from e
            raise

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
        """Invoke the Tool.run function and handle converting to an Output object.

        This function returns a tuple consisting of the output and an Output object, as expected by
        langchain tools. It captures the output (artifact) directly instead of serializing
        it to a string first.

        Args:
            ctx (ExecutionContext): The execution context for the tool.
            *args (Any): Additional positional arguments for the tool function.
            **kwargs (Any): Additional keyword arguments for the tool function.

        Returns:
            tuple[str, Output[SERIALIZABLE_TYPE_VAR]]: A tuple containing the output and the Output.

        """
        intermediate_output = self._run(ctx, *args, **kwargs)
        return (intermediate_output.value, intermediate_output)  # type: ignore  # noqa: PGH003

    def _generate_tool_description(self) -> str:
        """Generate tool descriptions.

        This function generates a comprehensive description of the tool, including its name,
        arguments, and output schema. The description is rendered using a Jinja template.

        Returns:
            str: The generated tool description in XML format.

        """
        args = []
        args_name_description_dict = []
        out_type = self.output_schema[0]
        out_description = self.output_schema[1]
        schema = self.args_json_schema()
        for arg, attribute in schema["properties"].items():
            arg_dict = {
                "name": arg,
                "type": attribute.get("type", None),
                "required": arg in schema.get("required", []),
            }
            args_name_description_dict.append(arg_dict)
            if "type" in attribute:
                args.append(f"{arg}: '{attribute['type']}'")

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
        """Check that the description is less than 1024 characters.

        OpenAI has a maximum function description length of 1024 characters. This validator
        ensures that the tool description does not exceed this limit.

        Returns:
            Self: The current instance of the tool.

        Raises:
            InvalidToolDescriptionError: If the description exceeds the maximum length.

        """
        description_length = len(self._generate_tool_description())
        if description_length > MAX_TOOL_DESCRIPTION_LENGTH:
            raise InvalidToolDescriptionError(self.name)
        return self

    def to_langchain(self, ctx: ExecutionContext) -> StructuredTool:
        """Return a LangChain representation of this tool.

        This function provides a LangChain-compatible version of the tool. The response format is
        the default one without including artifacts. The ExecutionContext is baked into the
        StructuredTool via a partial run function.

        Args:
            ctx (ExecutionContext): The execution context for the tool.

        Returns:
            StructuredTool: The LangChain-compatible representation of the tool, including the
            tool's name, description, and argument schema, with the execution context baked
            into the function.

        """
        return StructuredTool(
            name=self.name.replace(" ", "_"),
            description=self._generate_tool_description(),
            args_schema=self.args_schema,
            func=partial(self._run, ctx),
        )

    def to_langchain_with_artifact(self, ctx: ExecutionContext) -> StructuredTool:
        """Return a LangChain representation of this tool with content and artifact.

        This function provides a LangChain-compatible version of the tool, where the response format
        includes both the content and the artifact. The ExecutionContext is baked into the
        StructuredTool via a partial run function for capturing output directly.

        Args:
            ctx (ExecutionContext): The execution context for the tool.

        Returns:
            StructuredTool: The LangChain-compatible representation of the tool, including the
            tool's name, description, argument schema, and the ability to return both content
            and artifact.

        """
        return StructuredTool(
            name=self.name.replace(" ", "_"),
            description=self._generate_tool_description(),
            args_schema=self.args_schema,
            func=partial(self._run_with_artifacts, ctx),
            return_direct=True,
            response_format="content_and_artifact",
        )

    def args_json_schema(self) -> dict[str, Any]:
        """Return the json_schema for the tool args.

        This function retrieves the JSON schema for the tool's arguments, which defines the expected
        input structure.

        Returns:
            dict[str, Any]: The JSON schema representing the tool's arguments.

        """
        return self.args_schema.model_json_schema()

    def __str__(self) -> str:
        """Return the string representation.

        This method generates a string representation of the tool, including its ID, name,
        description, argument schema, and output schema.

        Returns:
            str: A string representation of the tool.

        """
        return (
            f"ToolModel(id={self.id!r}, name={self.name!r}, "
            f"description={self.description!r}, "
            f"args_schema={self.args_schema.__name__!r}, "
            f"output_schema={self.output_schema!r})"
        )

    @field_serializer("args_schema")
    def serialize_args_schema(self, value: type[BaseModel]) -> str:
        """Serialize the args_schema by returning its class name.

        This function serializes the arguments schema by returning the class name of the schema type.

        Args:
            value (type[BaseModel]): The argument schema class.

        Returns:
            str: The class name of the argument schema.

        """
        return value.__name__


class PortiaRemoteTool(Tool, Generic[SERIALIZABLE_TYPE_VAR]):
    """Tool that passes run execution to Portia Cloud."""

    api_key: SecretStr
    api_endpoint: str

    def parse_response(self, response: dict[str, Any]) -> Output:
        """Parse a JSON response into domain models or errors.

        This method handles the response from the Portia Cloud API, converting it into domain
        specific models. It also handles errors, including `ToolSoftError` and `ToolHardError`,
        as well as clarifications of different types.

        Args:
            response (dict[str, Any]): The JSON response returned by the Portia Cloud API.

        Returns:
            Output: The parsed output wrapped in an `Output` object.

        Raises:
            ToolSoftError: If a soft error is encountered in the response.
            ToolHardError: If a hard error is encountered in the response.

        """
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
                            id=clarification["id"],
                            action_url=HttpUrl(clarification["action_url"]),
                            user_guidance=clarification["user_guidance"],
                        ),
                    )
                case "Input Clarification":
                    return Output(
                        value=InputClarification(
                            id=clarification["id"],
                            argument_name=clarification["argument_name"],
                            user_guidance=clarification["user_guidance"],
                        ),
                    )
                case "Multiple Choice Clarification":
                    return Output(
                        value=MultipleChoiceClarification(
                            id=clarification["id"],
                            argument_name=clarification["argument_name"],
                            user_guidance=clarification["user_guidance"],
                            options=clarification["options"],
                        ),
                    )
                case "Value Confirmation Clarification":
                    return Output(
                        value=ValueConfirmationClarification(
                            id=clarification["id"],
                            argument_name=clarification["argument_name"],
                            user_guidance=clarification["user_guidance"],
                        ),
                    )
        return output

    def run(
        self,
        ctx: ExecutionContext,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  #  noqa: ANN401
    ) -> SERIALIZABLE_TYPE_VAR | None | Clarification:
        """Invoke the run endpoint and handle the response.

        This method sends the execution request to the Portia Cloud API, passing the arguments and
        execution context. It then processes the response by calling `parse_response`. Errors
        during the request or parsing are raised as `ToolHardError`.

        Args:
            ctx (ExecutionContext): The context of the execution, including end user ID, workflow ID
            and additional data.
            *args (Any): The positional arguments for the tool.
            **kwargs (Any): The keyword arguments for the tool.

        Returns:
            SERIALIZABLE_TYPE_VAR | None | Clarification: The result of the run execution, which
            could either be a serialized value, None, or a `Clarification` object.

        Raises:
            ToolHardError: If the request fails or there is an error parsing the response.

        """
        try:
            # Send to Cloud
            response = httpx.post(
                url=f"{self.api_endpoint}/api/v0/tools/{self.id}/run/",
                content=json.dumps(
                    {
                        "arguments": combine_args_kwargs(*args, **kwargs),
                        "execution_context": {
                            "end_user_id": ctx.end_user_id or "",
                            "workflow_id": ctx.workflow_id,
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
