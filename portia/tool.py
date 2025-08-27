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

import asyncio
import inspect
import json
from abc import abstractmethod
from datetime import timedelta
from functools import partial
from typing import Any, Generic, Self, TypeVar

import httpx
import mcp
from jsonref import replace_refs
from langchain_core.tools import StructuredTool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationError,
    field_serializer,
    model_validator,
)

from portia.clarification import (
    ActionClarification,
    Clarification,
    ClarificationCategory,
    ClarificationListType,
    ClarificationUUID,
    InputClarification,
    MultipleChoiceClarification,
    ValueConfirmationClarification,
)
from portia.cloud import PortiaCloudClient
from portia.common import SERIALIZABLE_TYPE_VAR, combine_args_kwargs
from portia.config import Config
from portia.end_user import EndUser
from portia.errors import InvalidToolDescriptionError, ToolHardError, ToolSoftError
from portia.execution_agents.execution_utils import is_clarification
from portia.execution_agents.output import LocalDataValue, Output
from portia.logger import logger
from portia.mcp_session import McpClientConfig, get_mcp_session
from portia.plan import Plan
from portia.plan_run import PlanRun
from portia.templates.render import render_template

"""MAX_TOOL_DESCRIPTION_LENGTH is limited to stop overflows in the planner context window."""
MAX_TOOL_DESCRIPTION_LENGTH = 16384


class ToolRunContext(BaseModel):
    """Context passed to tools when running.

    Attributes:
        plan_run(PlanRun): The run the tool run is part of.
        plan(Plan): The plan the tool run is part of.
        config(Config): The config for the SDK as a whole.
        clarifications(ClarificationListType): Relevant clarifications for this tool plan_run.

    """

    model_config = ConfigDict(extra="forbid")

    end_user: EndUser
    plan_run: PlanRun
    plan: Plan
    config: Config
    clarifications: ClarificationListType


class _ArgsSchemaPlaceholder(BaseModel):
    """Placeholder ArgsSchema for tools that take no arguments."""


class ReadyResponse(BaseModel):
    """Response from the /ready endpoint."""

    ready: bool
    clarifications: ClarificationListType


class Tool(BaseModel, Generic[SERIALIZABLE_TYPE_VAR]):
    """Abstract base class for a tool.

    This class serves as the blueprint for all tools. Child classes must implement the `run` method.

    Attributes:
    id (str): A unique identifier for the tool.
        This must be unique as collisions in a tool registry will lead to errors.
    name (str): The name of the tool. The name is informational only but useful for debugging.
    description (str): Purpose of the tool and usage.
        This is important information for the planning_agent module to know when and
        how to use this tool.
    args_schema (type[BaseModel]): The schema defining the expected input arguments for the tool.
        We use Pydantic models to define these types.
    output_schema (tuple[str, str]): A tuple containing the type and description of the tool's
        output. To maximize the advantages of using an agentic approach this doesn't need to be
        tightly defined. Instead it should give just a high level overview of the type and
        contents of the tools output.
    should_summarize (bool): Indicates whether the tool's output should be automatically summarized
        by the summarizer agent. For some tools summarization is useful (for example: a tool
        that fetches the latest news) whereas other tools it's not (for example: a tool
        that fetches raw price data).

    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

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
    structured_output_schema: type[BaseModel] | None = Field(
        default=None,
        description="The schema to use for the structured output of the tool if summarization is "
        "enabled. If not provided, the output will be the default output schema of the tool.run() "
        "method.",
    )

    def ready(self, ctx: ToolRunContext) -> ReadyResponse:  # noqa: ARG002
        """Check whether the tool can be plan_run.

        This method can be implemented by subclasses to allow checking if the tool can be plan_run.
        It may run any authentication logic or other required checks before returning its status.
        If left unimplemented will always return true.

        Args:
            ctx (ToolRunContext): Co
            ntext of the tool run

        Returns:
            ReadyResponse: Whether the tool is ready to run and any clarifications that need to be
            resolved

        """
        return ReadyResponse(ready=True, clarifications=[])

    @abstractmethod
    def run(
        self,
        ctx: ToolRunContext,
        *args: Any,
        **kwargs: Any,
    ) -> SERIALIZABLE_TYPE_VAR | Clarification:
        """Run the tool.

        This method must be implemented by subclasses to define the tool's specific behavior.

        Args:
            ctx (ToolRunContext): Context of the tool execution
            args (Any): The arguments passed to the tool for execution.
            kwargs (Any): The keyword arguments passed to the tool for execution.

        Returns:
            Any: The result of the tool's execution which can be any serializable type
            or a clarification.

        """

    def _run(
        self,
        ctx: ToolRunContext,
        *args: Any,
        **kwargs: Any,
    ) -> Output:
        """Invoke the Tool.run function and handle converting the result into an Output object.

        This is the entry point for agents to invoke a tool.

        Args:
            ctx (ToolRunContext): The context for the tool.
            *args (Any): Additional positional arguments for the tool function.
            **kwargs (Any): Additional keyword arguments for the tool function.

        Returns:
            Output: The tool's output wrapped in an Output object.

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
        return self._parse_output(output)

    async def arun(
        self,
        ctx: ToolRunContext,
        *args: Any,
        **kwargs: Any,
    ) -> SERIALIZABLE_TYPE_VAR | Clarification:
        """Async run the tool.

        This is the method that tools should implement if they want to provide an async method for
        running the tool. When calling this method, you should likely call the _arun method instead.

        Args:
            ctx (ToolRunContext): The context for the tool.
            *args (Any): Additional positional arguments for the tool function.
            **kwargs (Any): Additional keyword arguments for the tool function.

        Returns:
            SERIALIZABLE_TYPE_VAR | Clarification: The result of the tool's execution which can be
            any serializable type or a clarification.

        """
        raise NotImplementedError("Async run is not implemented")  # pragma: no cover

    async def _arun(
        self,
        ctx: ToolRunContext,
        *args: Any,
        **kwargs: Any,
    ) -> Output:
        """Async run the tool.

        This method must be implemented by subclasses to define the tool's specific behavior.
        This is the method that should be called internally by anything calling a tool.
        """
        try:
            output = await self.arun(ctx, *args, **kwargs)
        except NotImplementedError:
            # if the subclass does not implement arun, we can just call the sync run method
            return await asyncio.to_thread(self._run, ctx, *args, **kwargs)
        except Exception as e:
            # check if error is wrapped as a Hard or Soft Tool Error.
            # if not wrap as ToolSoftError
            if not isinstance(e, ToolHardError) and not isinstance(e, ToolSoftError):
                raise ToolSoftError(e) from e
            raise
        return self._parse_output(output)

    def _parse_output(self, output: SERIALIZABLE_TYPE_VAR | Clarification) -> Output:
        """Parse the output of the tool.

        This method handles the output of the tool and converts it to an Output object.
        """
        if is_clarification(output):
            clarifications = output if isinstance(output, list) else [output]
            return LocalDataValue(
                value=clarifications,
            )
        if self.structured_output_schema:
            # try to coerce output to structured output schema if it's not already, but fall back
            # to the default output schema, letting the llm step summarizer handle it
            if isinstance(output, self.structured_output_schema):
                return LocalDataValue(value=output)
            try:
                return LocalDataValue(value=self.structured_output_schema.model_validate(output))
            except ValidationError:
                pass
        return LocalDataValue(value=output)  # type: ignore  # noqa: PGH003

    def _run_with_artifacts(
        self,
        ctx: ToolRunContext,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, Output]:
        """Invoke the Tool.run function and handle converting to an Output object.

        This function returns a tuple consisting of the output and an Output object, as expected by
        langchain tools. It captures the output (artifact) directly instead of serializing
        it to a string first.

        Args:
            ctx (ToolRunContext): The context for the tool.
            *args (Any): Additional positional arguments for the tool function.
            **kwargs (Any): Additional keyword arguments for the tool function.

        Returns:
            tuple[str, Output]: A tuple containing the output and the Output.

        """
        intermediate_output = self._run(ctx, *args, **kwargs)
        return (intermediate_output.get_value(), intermediate_output)  # type: ignore  # noqa: PGH003

    async def _arun_with_artifacts(
        self,
        ctx: ToolRunContext,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, Output]:
        """Async run the tool with artifacts.

        This function returns a tuple consisting of the output and an Output object, as expected by
        langchain tools. It captures the output (artifact) directly instead of serializing
        it to a string first.

        Args:
            ctx (ToolRunContext): The context for the tool.
            *args (Any): Additional positional arguments for the tool function.
            **kwargs (Any): Additional keyword arguments for the tool function.

        Returns:
            tuple[str, Output]: A tuple containing the output and the Output.

        """
        intermediate_output = await self._arun(ctx, *args, **kwargs)
        return (intermediate_output.get_value(), intermediate_output)  # type: ignore  # noqa: PGH003

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
            if attribute.get("enum", None):
                arg_dict["enum"] = attribute.get("enum")
            if attribute.get("default", None):
                arg_dict["default"] = attribute.get("default")

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
        """Check that the description is less than 16384 characters.

        OpenAI has a maximum function description length of 16384 characters. This validator
        ensures that the tool description does not exceed this limit.

        Returns:
            Self: The current instance of the tool.

        Raises:
            InvalidToolDescriptionError: If the description exceeds the maximum length.

        """
        description_length = len(self._generate_tool_description())
        if description_length > MAX_TOOL_DESCRIPTION_LENGTH:
            raise InvalidToolDescriptionError(self.id)
        return self

    @model_validator(mode="after")
    def check_run_method_signature(self) -> Self:
        """Ensure the run method signature matches the args_schema."""
        try:
            sig = inspect.signature(self.__class__.run, eval_str=True)
        except NameError:
            # Dont fail if the types cannot be extracted. This can happen with eval_str=True
            # if the class is not defined in a global scope. Since this validator is only
            # for warnings we can just exit here.
            return self

        params = list(sig.parameters.values())

        if params and params[0].name == "self":
            params = params[1:]
        if not params:
            return self

        ctx_param = params[0]
        if ctx_param.annotation not in (
            ToolRunContext,
            inspect.Signature.empty,
        ):
            logger().warning("First argument of run must be annotated as ToolRunContext")

        param_map = {
            p.name: p.annotation
            for p in params[1:]
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

        for arg_name, arg_annotation in param_map.items():
            pydantic_field = self.args_schema.model_fields.get(arg_name)
            if pydantic_field is None:
                logger().warning(f"Unknown argument '{arg_name}' in run method")
            elif (
                arg_annotation is not inspect.Signature.empty
                and arg_annotation != pydantic_field.annotation
            ):
                logger().warning(
                    f"Run method argument '{arg_name}' type {arg_annotation} does not match "
                    f"args_schema field type: {pydantic_field.annotation}"
                )

        return self

    def to_langchain(self, ctx: ToolRunContext, sync: bool = True) -> StructuredTool:
        """Return a LangChain representation of this tool.

        This function provides a LangChain-compatible version of the tool. The response format is
        the default one without including artifacts. The ExecutionContext is baked into the
        StructuredTool via a partial run function.

        Args:
            ctx (ToolRunContext): The context for the tool.
            sync (bool): Whether to use the sync or async version of the tool.

        Returns:
            StructuredTool: The LangChain-compatible representation of the tool, including the
            tool's name, description, and argument schema, with the execution context baked
            into the function.

        """
        return (
            StructuredTool(
                name=self.name.replace(" ", "_"),
                description=self._generate_tool_description(),
                args_schema=self.args_schema,
                func=partial(self._run, ctx),
                return_direct=True,
            )
            if sync
            else StructuredTool(
                name=self.name.replace(" ", "_"),
                description=self._generate_tool_description(),
                args_schema=self.args_schema,
                coroutine=partial(self._arun, ctx),
                return_direct=True,
            )
        )

    def to_langchain_with_artifact(self, ctx: ToolRunContext, sync: bool = True) -> StructuredTool:
        """Return a LangChain representation of this tool with content and artifact.

        This function provides a LangChain-compatible version of the tool, where the response format
        includes both the content and the artifact. The ToolRunContext is baked into the
        StructuredTool via a partial run function for capturing output directly.

        Args:
            ctx (ToolRunContext): The context for the tool.
            sync (bool): Whether to use the sync or async version of the tool.

        Returns:
            StructuredTool: The LangChain-compatible representation of the tool, including the
            tool's name, description, argument schema, and the ability to return both content
            and artifact.

        """
        return (
            StructuredTool(
                name=self.name.replace(" ", "_"),
                description=self._generate_tool_description(),
                args_schema=self.args_schema,
                func=partial(self._run_with_artifacts, ctx),
                return_direct=True,
                response_format="content_and_artifact",
            )
            if sync
            else StructuredTool(
                name=self.name.replace(" ", "_"),
                description=self._generate_tool_description(),
                args_schema=self.args_schema,
                coroutine=partial(self._arun_with_artifacts, ctx),
                return_direct=True,
                response_format="content_and_artifact",
            )
        )

    def args_json_schema(self) -> dict[str, Any]:
        """Return the json_schema for the tool args.

        This function retrieves the JSON schema for the tool's arguments, which defines the expected
        input structure.

        Returns:
            dict[str, Any]: The JSON schema representing the tool's arguments.

        """
        return replace_refs(self.args_schema.model_json_schema())  # type: ignore  # noqa: PGH003

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

        This function serializes the arguments schema by returning the class name of the schema.

        Args:
            value (type[BaseModel]): The argument schema class.

        Returns:
            str: The class name of the argument schema.

        """
        return value.__name__

    def pretty(self) -> str:
        """Return a pretty string representation of the tool."""
        title = f"| {self.name} ({self.id}) |"
        return (
            f"{'-' * len(title)}\n{title}\n{'-' * len(title)}"
            f"\n{self._generate_tool_description()}"
        )


class PortiaRemoteTool(Tool, Generic[SERIALIZABLE_TYPE_VAR]):
    """Tool that passes run execution to Portia Cloud."""

    client: httpx.Client

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse_response(self, ctx: ToolRunContext, response: dict[str, Any]) -> Output:
        """Parse a JSON response into domain models or errors.

        This method handles the response from the Portia Cloud API, converting it into domain
        specific models. It also handles errors, including `ToolSoftError` and `ToolHardError`,
        as well as clarifications of different types.

        Args:
            ctx (ToolRunContext): Context of the environment
            response (dict[str, Any]): The JSON response returned by the Portia Cloud API.

        Returns:
            Output: The parsed output wrapped in an `Output` object.

        Raises:
            ToolSoftError: If a soft error is encountered in the response.
            ToolHardError: If a hard error is encountered in the response.

        """
        output = LocalDataValue.model_validate(response["output"])
        output_value = output.get_value()

        # Handle Tool Errors
        if response.get("soft_error", False):
            raise ToolSoftError(str(output_value))

        # Handle Clarifications
        if isinstance(output_value, list) and output_value and "category" in output_value[0]:
            clarification = output_value[0]
            match clarification["category"]:
                case ClarificationCategory.ACTION:
                    return LocalDataValue(
                        value=ActionClarification(
                            plan_run_id=ctx.plan_run.id,
                            id=ClarificationUUID.from_string(clarification["id"]),
                            action_url=HttpUrl(clarification["action_url"]),
                            user_guidance=clarification["user_guidance"],
                            source="Portia remote tool",
                        ),
                    )
                case ClarificationCategory.INPUT:
                    return LocalDataValue(
                        value=InputClarification(
                            plan_run_id=ctx.plan_run.id,
                            id=ClarificationUUID.from_string(clarification["id"]),
                            argument_name=clarification["argument_name"],
                            user_guidance=clarification["user_guidance"],
                            source="Portia remote tool",
                        ),
                    )
                case ClarificationCategory.MULTIPLE_CHOICE:
                    return LocalDataValue(
                        value=MultipleChoiceClarification(
                            plan_run_id=ctx.plan_run.id,
                            id=ClarificationUUID.from_string(clarification["id"]),
                            argument_name=clarification["argument_name"],
                            user_guidance=clarification["user_guidance"],
                            options=clarification["options"],
                            source="Portia remote tool",
                        ),
                    )
                case ClarificationCategory.VALUE_CONFIRMATION:
                    return LocalDataValue(
                        value=ValueConfirmationClarification(
                            plan_run_id=ctx.plan_run.id,
                            id=ClarificationUUID.from_string(clarification["id"]),
                            argument_name=clarification["argument_name"],
                            user_guidance=clarification["user_guidance"],
                            source="Portia remote tool",
                        ),
                    )
        return output

    def ready(self, ctx: ToolRunContext) -> ReadyResponse:
        """Check if the remote tool is ready by calling the /ready endpoint.

        Args:
            ctx (ToolRunContext): Context of the environment

        Returns:
            ReadyResponse: Whether the tool is ready to run and any clarifications that
              need to be resolved

        """
        try:
            # Send to Cloud
            response = self.client.post(
                url=f"/api/v0/tools/{self.id}/ready/",
                content=json.dumps(
                    {
                        "execution_context": {
                            "end_user_id": ctx.end_user.external_id,
                            "plan_run_id": str(ctx.plan_run.id),
                        },
                    },
                ),
            )
            response.raise_for_status()
        except Exception as e:  # noqa: BLE001
            logger().error(f"Unhandled error from Portia Cloud: {e}")
            return ReadyResponse(ready=False, clarifications=[])
        else:
            response_json = response.json()
            try:
                ready = ReadyResponse.model_validate(response_json)
            except ValidationError:
                # Old format response
                return ReadyResponse(
                    ready="success" in response_json,
                    clarifications=response_json.get("clarifications", []),
                )
            else:
                return ready

    def run(
        self,
        ctx: ToolRunContext,
        *args: Any,
        **kwargs: Any,
    ) -> SERIALIZABLE_TYPE_VAR | None | Clarification:
        """Invoke the run endpoint and handle the response.

        This method sends the execution request to the Portia Cloud API, passing the arguments and
        execution context. It then processes the response by calling `parse_response`. Errors
        during the request or parsing are raised as `ToolHardError`.

        Args:
            ctx (ToolRunContext): The context of the execution, including end user ID, run ID
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
            # Default function for JSON serialization of Pydantic models
            def default_serializer(
                obj: Any,  # noqa: ANN401
            ) -> dict[str, Any] | list[Any] | str | int | float | bool | None:
                if isinstance(obj, BaseModel):
                    return json.loads(obj.model_dump_json())
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")  # noqa: TRY301

            # Send to Cloud
            response = self.client.post(
                url=f"/api/v0/tools/{self.id}/run/",
                content=json.dumps(
                    {
                        "arguments": combine_args_kwargs(*args, **kwargs),
                        "execution_context": {
                            "end_user_id": ctx.end_user.external_id,
                            "plan_run_id": str(ctx.plan_run.id),
                            "additional_data": ctx.end_user.additional_data,
                        },
                    },
                    default=default_serializer,
                ),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger().error(f"Error from Portia Cloud: {e.response.content}")
            raise ToolHardError(str(e.response.json())) from e
        except Exception as e:
            logger().error(f"Unhandled error from Portia Cloud: {e}")
            raise ToolHardError(e) from e
        else:
            try:
                output = self.parse_response(ctx, response.json())
            except (ValidationError, KeyError) as e:
                logger().error(f"Error parsing response from Portia Cloud: {e}")
                raise ToolHardError(e) from e
            else:
                return output.get_value()

    @classmethod
    def batch_ready_check(
        cls,
        config: Config,
        tool_ids: set[str],
        tool_run_context: ToolRunContext,
    ) -> ReadyResponse:
        """Batch check readiness for Portia cloud tools.

        Args:
            config (Config): The config for the SDK as a whole.
            tool_ids (set[str]): The list of tool IDs to check readiness for.
            tool_run_context (ToolRunContext): The context of the execution.

        Returns:
            ReadyResponse: The readiness response for the tools.

        """
        client = PortiaCloudClient.new_client(config)
        logger().debug("Checking readiness for Portia cloud tools: " + ", ".join(tool_ids))
        batch_ready_response = client.post(
            url="/api/v0/tools/batch/ready/",
            json={
                "tool_ids": sorted(tool_ids),
                "execution_context": {
                    "end_user_id": tool_run_context.end_user.external_id,
                    "plan_run_id": str(tool_run_context.plan_run.id),
                },
            },
        )
        if batch_ready_response.status_code == httpx.codes.NOT_FOUND:
            # For backwards compatibility: if the endpoint is not found, we set ready=true
            # and fallback to individual tool ready checks failing.
            return ReadyResponse(ready=True, clarifications=[])
        batch_ready_response.raise_for_status()
        return ReadyResponse.model_validate(batch_ready_response.json())


class PortiaMcpTool(Tool[str]):
    """A Portia Tool wrapper for an MCP server-based tool."""

    mcp_client_config: McpClientConfig

    def run(self, _: ToolRunContext, **kwargs: Any) -> str:
        """Invoke the tool by dispatching to the MCP server.

        Args:
            _: The tool run context
            **kwargs: The arguments to pass to the MCP tool invocation

        Returns:
            str: The result of the tool call

        """
        logger().debug(f"Calling tool {self.name} with arguments {kwargs}")
        return asyncio.run(self.call_remote_mcp_tool(self.name, kwargs))

    async def arun(self, _: ToolRunContext, **kwargs: Any) -> str:
        """Invoke the tool by dispatching to the MCP server asynchronously."""
        return await self.call_remote_mcp_tool(self.name, kwargs)

    async def call_remote_mcp_tool(self, name: str, arguments: dict | None = None) -> str:
        """Call a tool using the MCP session.

        There are issues with the implementation of the mcp client which mean that the
        `read_timeout_seconds` still waits for a response from the server before raising
        a timeout, which is entirely defeating the purpose of the timeout on our side.

        This method implements a custom timeout using `asyncio.wait`, allowing us to
        raise the correct exception when the deadline is reached.
        """
        task = asyncio.create_task(self._call_mcp_tool(name, arguments))
        done, _ = await asyncio.wait(
            [task],
            timeout=self.mcp_client_config.tool_call_timeout_seconds,
        )
        if task not in done:
            task.cancel()
            raise ToolSoftError(
                "MCP tool timed out after "
                f"{self.mcp_client_config.tool_call_timeout_seconds}s: "
                f"{self.name}({self.id})"
            )
        return self._handle_mcp_tool_result(task)

    def _handle_mcp_tool_result(self, task: asyncio.Task[str]) -> str:
        """Handle the result of a tool call.

        Handles the ExceptionGroup structure that come from the MCP client,
        unpacking them into ToolSoftError and ToolHardError.

        ExceptionGroups have to be fully consumed in order to not raise another
        ExceptionGroup.

        E.G.
        ```
        try:
            raise ExceptionGroup("test", [ValueError("test"), TypeError("test2")])
        except* ValueError as eg:
            raise CustomError() from eg
        ```
        This code will still raise an ExceptionGroup, because the `TypeError` is not
        consumed by the `except*` blocks.

        Catching `except* Exception` will consume all exceptions in the group,
        so the following code will raise a `CustomError`:
        ```
        try:
            raise ExceptionGroup("test", [ValueError("test"), TypeError("test2")])
        except* Exception as eg:
            raise CustomError() from eg
        ```
        """
        try:
            return task.result()
        except* Exception as eg:
            # Distinguish timeouts from other MCP errors using the error code
            for inner in flatten_exceptions(eg, mcp.McpError):
                # REQUEST_TIMEOUT is raised by the MCP client on per-request timeouts
                if inner.error.code == httpx.codes.REQUEST_TIMEOUT:
                    raise ToolSoftError(
                        "MCP tool timed out after "
                        f"{self.mcp_client_config.tool_call_timeout_seconds}s: "
                        f"{self.name}({self.id})"
                    ) from None
            # Non-timeout MCP errors: surface as a soft error for callers
            for inner in flatten_exceptions(eg, ToolHardError):
                raise inner from None
            raise ToolHardError(
                f"MCP tool {self.name}({self.id}) error: {flatten_exceptions(eg, Exception)}"
            ) from eg

    async def _call_mcp_tool(self, name: str, arguments: dict | None = None) -> str:
        """Call a tool using the MCP session."""
        async with get_mcp_session(self.mcp_client_config) as session:
            tool_result = await session.call_tool(
                name,
                arguments,
                read_timeout_seconds=(
                    timedelta(seconds=self.mcp_client_config.tool_call_timeout_seconds)
                    if self.mcp_client_config.tool_call_timeout_seconds
                    else None
                ),
            )
            if tool_result.isError:
                raise ToolHardError(
                    f"MCP tool {self.name}({self.id}) returned an error: "
                    f"{tool_result.model_dump_json()}"
                )
            return tool_result.model_dump_json()


ExceptionT = TypeVar("ExceptionT", bound=BaseException)


def flatten_exceptions(
    exc_group: BaseExceptionGroup[Any], exc_type: type[ExceptionT]
) -> list[ExceptionT]:
    """Flatten an ExceptionGroup into a list of exceptions of a given type."""
    result = []
    for exc in exc_group.exceptions:
        if isinstance(exc, ExceptionGroup):
            result.extend(flatten_exceptions(exc, exc_type))
        elif isinstance(exc, exc_type):
            result.append(exc)
    return result
