"""Browser tools.

This module contains tools that can be used to navigate to a URL, authenticate the user,
and complete tasks.

The browser tool can run locally or using [Browserbase](https://browserbase.com/). If using
Browserbase, a Browserbase API key is required and project ID is required, and the tool can handle
separate end user authentication.

The browser tool can be used to navigate to a URL and complete tasks. If authentication is
required, the tool will return an ActionClarification with the user guidance and login URL.
If authentication is not required, the tool will return the task output. It uses
(BrowserUse)[https://browser-use.com/] for the task navigation.

"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from browser_use import Agent, Browser, BrowserConfig, Controller
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from pydantic_core import PydanticUndefined

from portia import logger
from portia.clarification import ActionClarification
from portia.common import validate_extras_dependencies
from portia.errors import ToolHardError
from portia.model import GenerativeModel  # noqa: TC001 - used in Pydantic Schema
from portia.tool import Tool, ToolRunContext

if TYPE_CHECKING:
    from browserbase.types import SessionCreateResponse

    from portia import Plan, PlanRun

NotSet: Any = PydanticUndefined

BROWSERBASE_AVAILABLE = validate_extras_dependencies("tools-browser-browserbase", raise_error=False)

T = TypeVar("T", bound=str | BaseModel)
BM = TypeVar("BM", bound=BaseModel)


class BrowserToolForUrlSchema(BaseModel):
    """Input schema for the BrowserToolForUrl.

    This schema defines the expected input parameters for the BrowserToolForUrl class.

    Attributes:
        task (str): The task description that should be performed by the browser tool.
            This is a required field that specifies what actions should be taken
            on the predefined URL.
        task_data (list[Any] | str | None): Task data that should be used to complete the task.
            Can be a string, a list of strings, or a list of objects that will be converted to
            strings. Important: This should include all relevant data in their entirety,
            from the first to the last character (i.e. NOT a summary).

    """

    task: str = Field(
        ...,
        description="The task to be completed by the Browser tool.",
    )
    task_data: list[Any] | str | None = Field(
        default=None,
        description="Task data that should be used to complete the task. "
        "Can be a string, a list of strings, "
        "or a list of objects that will be converted to strings. "
        "Important: This should include all relevant data in their entirety, "
        "from the first to the last character (i.e. NOT a summary).",
    )


class BrowserToolSchema(BaseModel):
    """Input schema for the BrowserTool.

    This schema defines the expected input parameters for the BrowserTool class.

    Attributes:
        url (str): The URL that the browser tool should navigate to.
            This is a required field specifying the target webpage.
        task (str): The task description that should be performed by the browser tool.
            This is a required field that specifies what actions should be taken
            on the provided URL.
        task_data (list[Any] | str | None): Task data that should be used to complete the task.
            Can be a string, a list of strings, or a list of objects that will be converted to
            strings. Important: This should include all relevant data in their entirety,
            from the first to the last character (i.e. NOT a summary).

    """

    url: str = Field(
        ...,
        description="The URL to navigate to.",
    )
    task: str = Field(
        ...,
        description="The task to be completed by the Browser tool.",
    )
    task_data: list[Any] | str | None = Field(
        default=None,
        description="Task data that should be used to complete the task. "
        "Can be a string, a list of strings, "
        "or a list of objects that will be converted to strings. "
        "Important: This should include all relevant data in their entirety, "
        "from the first to the last character (i.e. NOT a summary).",
    )


class BrowserTaskOutput(BaseModel, Generic[T]):
    """Output schema for browser task execution.

    This class represents the response from executing a browser task,
    including both the task result and any authentication requirements.

    Attributes:
        task_output (T): The result or output from executing the requested task.
        human_login_required (bool): Indicates if manual user authentication is needed.
            Defaults to False.
        login_url (str, optional): The URL where the user needs to go to authenticate.
            Only provided when human_login_required is True.
        user_login_guidance (str, optional): Instructions for the user on how to complete
            the login process. Only provided when human_login_required is True.

    """

    task_output: T | None = Field(
        default=None,
        description="The output from the task. `None` if authentication is required.",
    )
    human_login_required: bool = Field(
        default=False,
        description="Whether the user needs to login to complete the task.",
    )
    login_url: str | None = Field(
        default=None,
        description="The URL to navigate to for login if the user is not authenticated.",
    )
    user_login_guidance: str | None = Field(
        default=None,
        description="Guidance for the user to login if they are not authenticated.",
    )


class BrowserInfrastructureOption(Enum):
    """Enumeration of supported browser infrastructure providers.

    This enum defines the available options for running browser automation tasks.

    Attributes:
        LOCAL: Uses a local Chrome browser instance for automation.
            Suitable for development and testing.
        BROWSERBASE: Uses the Browserbase cloud service for automation.
            Provides better scalability and isolation between users.

    """

    LOCAL = "local"
    REMOTE = "remote"


class BrowserTool(Tool[str | BaseModel]):
    """General purpose browser tool. Customizable to user requirements.

    This tool is designed to be used for tasks that require a browser. If authentication is
    required, the tool will return an ActionClarification with the user guidance and login URL.
    If authentication is not required, the tool will return the task output. It uses
    (BrowserUse)[https://browser-use.com/] for the task navigation.

    When using the tool, you should ensure that once the user has authenticated, that they
    indicate that authentication is completed and resume the plan run.

    The tool supports both local and BrowserBase infrastructure providers for running the web
    based tasks. If using local, a local Chrome instance will be used, and the tool will not
    support end_user_id. If using BrowserBase, a BrowserBase API key is required and the tool
    can handle separate end users. The infrastructure provider can be specified using the
    `infrastructure_option` argument.

    Args:
        id (str, optional): Custom identifier for the tool. Defaults to "browser_tool".
        name (str, optional): Display name for the tool. Defaults to "Browser Tool".
        description (str, optional): Custom description of the tool's purpose. Defaults to a
            general description of the browser tool's capabilities.
        infrastructure_option (BrowserInfrastructureOption, optional): The infrastructure
            provider to use. Can be either `BrowserInfrastructureOption.LOCAL` or
            `BrowserInfrastructureOption.REMOTE`. Defaults to
            `BrowserInfrastructureOption.REMOTE`.
        custom_infrastructure_provider (BrowserInfrastructureProvider, optional): A custom
            infrastructure provider to use. If not provided, the infrastructure provider will be
            resolved from the `infrastructure_option` argument.
        structured_output_schema (BaseModel, optional): A Pydantic model to use for structured
            output. If not provided, the tool will return a string.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(init_var=True, default="browser_tool")
    name: str = Field(init_var=True, default="Browser Tool")
    description: str = Field(
        init_var=True,
        default=(
            "General purpose browser tool. Can be used to navigate to a URL and complete tasks. "
            "Should only be used if the task requires a browser and you are sure of the URL. "
            "This tool handles a full end to end task. It is capable of doing multiple things "
            "across different URLs within the same root domain as part of the end to end task. As "
            "a result, do not call this tool more than once back to back unless it is for "
            "different root domains - just call it once with the combined task and the URL set "
            "to the root domain."
        ),
    )
    args_schema: type[BaseModel] = Field(init_var=True, default=BrowserToolSchema)
    output_schema: tuple[str, str] = ("str", "The Browser tool's response to the user query.")

    model: GenerativeModel | None | str = Field(
        default=None,
        exclude=True,
        description="The model to use for the BrowserTool. If not provided, "
        "the model will be resolved from the config.",
    )

    infrastructure_option: BrowserInfrastructureOption = Field(
        default_factory=lambda: BrowserInfrastructureOption.REMOTE
        if (
            BROWSERBASE_AVAILABLE
            and os.getenv("BROWSERBASE_API_KEY")
            and os.getenv("BROWSERBASE_PROJECT_ID")
        )
        else BrowserInfrastructureOption.LOCAL,
        description="The infrastructure provider to use for the browser tool.",
    )

    custom_infrastructure_provider: BrowserInfrastructureProvider | None = Field(default=None)

    structured_output_schema: type[BaseModel] | None = Field(
        default=None,
        description="Optional structured output schema for the browser tool's task output.",
    )

    @cached_property
    def infrastructure_provider(self) -> BrowserInfrastructureProvider:
        """Get the infrastructure provider instance (cached)."""
        if self.custom_infrastructure_provider:
            return self.custom_infrastructure_provider
        if self.infrastructure_option == BrowserInfrastructureOption.REMOTE:
            # Ensure error is raised if not installed
            validate_extras_dependencies("tools-browser-browserbase", raise_error=True)
            return BrowserInfrastructureProviderBrowserBase()
        return BrowserInfrastructureProviderLocal()

    @staticmethod
    def process_task_data(task_data: list[Any] | str | None) -> str:
        """Process task_data into a string, handling different input types.

        Args:
            task_data: Data that can be a None, a string or a list of objects.

        Returns:
            A string representation of the data, with list items joined by newlines.

        """
        if task_data is None:
            return ""

        if isinstance(task_data, str):
            return task_data

        return "\n".join(str(item) for item in task_data)

    def run(
        self, ctx: ToolRunContext, url: str, task: str, task_data: list[Any] | str | None = None
    ) -> str | BaseModel | ActionClarification:
        """Run the BrowserTool."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:  # pragma: no cover
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._run_browser_tasks(ctx, url, task, task_data))

    async def arun(
        self,
        ctx: ToolRunContext,
        url: str,
        task: str,
        task_data: list[Any] | str | None = None,
    ) -> str | BaseModel | ActionClarification:
        """Run the BrowserTool asynchronously."""
        return await self._run_browser_tasks(ctx, url, task, task_data)

    async def _run_browser_tasks(
        self, ctx: ToolRunContext, url: str, task: str, task_data: list[Any] | str | None = None
    ) -> str | BaseModel | ActionClarification:
        output_schema = self.structured_output_schema if self.structured_output_schema else str
        output_model = BrowserTaskOutput[output_schema]
        # Main task
        task_to_complete = (
            f"Go to {url} and complete the following task: {task}. The user may already be "
            "logged in. If the user is NOT already logged in and at any point login is "
            "required to complete the task, please return human_login_required=True, and the "
            "url of the sign in page as well as what the user should do to sign in. "
            "Additional data is available below: "
            f"{self.process_task_data(task_data)}"
        )

        task_result = await self._run_agent_task(ctx, task_to_complete, output_model)
        if task_result.human_login_required:
            return self._handle_login_requirement(ctx, task_result)

        self.infrastructure_provider.step_complete(ctx)
        if task_result.task_output is None:
            raise ToolHardError("Expected task output to be a string, got None")
        return task_result.task_output

    async def _run_agent_task(
        self,
        ctx: ToolRunContext,
        task_description: str,
        output_model: type[BM],
    ) -> BM:
        model = ctx.config.get_generative_model(self.model) or ctx.config.get_default_model()
        llm = model.to_langchain()
        agent = Agent(
            task=task_description,
            llm=llm,
            browser=self.infrastructure_provider.setup_browser(ctx),
            controller=Controller(output_model=output_model),
        )
        result = await agent.run()
        final_result = result.final_result()
        if not isinstance(final_result, str):
            raise ToolHardError(f"Expected final result to be a string, got {type(final_result)}")
        return output_model.model_validate(json.loads(final_result))

    def _handle_login_requirement(
        self,
        ctx: ToolRunContext,
        result: BrowserTaskOutput,
    ) -> ActionClarification:
        """Handle cases where login is required with an ActionClarification."""
        if result.user_login_guidance is None or result.login_url is None:
            raise ToolHardError(
                "Expected user guidance and login URL if human login is required",
            )
        return ActionClarification(
            user_guidance=result.user_login_guidance,
            action_url=self.infrastructure_provider.construct_auth_clarification_url(
                ctx,
                result.login_url,
            ),
            plan_run_id=ctx.plan_run.id,
            require_confirmation=True,
            source="Browser tool",
        )


class BrowserToolForUrl(BrowserTool):
    """Browser tool for a specific URL.

    This tool is designed to be used for browser-based tasks on the specified URL.
    If authentication is required, the tool will return an ActionClarification with the user
    guidance and login URL. If authentication is not required, the tool will return the task
    output. It uses (BrowserUse)[https://browser-use.com/] for the task navigation.

    When using the tool, the developer should ensure that once the user has completed
    authentication, that they resume the plan run.

    The tool supports both local and BrowserBase infrastructure providers for running the web
    based tasks. If using local, a local Chrome instance will be used, and the tool will not
    support end_user_id. If using BrowserBase, a BrowserBase API key is required and the tool
    can handle separate end users. The infrastructure provider can be specified using the
    `infrastructure_option` argument.

    Args:
        url (str): The URL that this browser tool will navigate to for all tasks.
        id (str, optional): Custom identifier for the tool. If not provided, will be generated
            based on the URL's domain.
        name (str, optional): Display name for the tool. If not provided, will be generated
            based on the URL's domain.
        description (str, optional): Custom description of the tool's purpose. If not provided,
            will be generated with the URL.
        infrastructure_option (BrowserInfrastructureOption, optional): The infrastructure
            provider to use. Can be either `BrowserInfrastructureOption.LOCAL` or
            `BrowserInfrastructureOption.REMOTE`. Defaults to
            `BrowserInfrastructureOption.REMOTE`.
        custom_infrastructure_provider (BrowserInfrastructureProvider, optional): A custom
            infrastructure provider to use. If not provided, the infrastructure provider will be
            resolved from the `infrastructure_option` argument.

    """

    url: str = Field(
        ...,
        description="The URL to navigate to.",
    )

    def __init__(
        self,
        url: str,
        id: str | None = None,  # noqa: A002
        name: str | None = None,
        description: str | None = None,
        model: GenerativeModel | None | str = NotSet,
        infrastructure_option: BrowserInfrastructureOption | None = NotSet,
    ) -> None:
        """Initialize the BrowserToolForUrl."""
        domain_parts = str(HttpUrl(url).host).split(".")
        formatted_domain = "_".join(domain_parts)
        if not id:
            id = f"browser_tool_for_url_{formatted_domain}"  # noqa: A001
        if not name:
            name = f"Browser Tool for {formatted_domain}"
        if not description:
            description = (
                f"Browser tool for the URL {url}. Can be used to navigate to the URL and complete "
                "tasks."
            )
        super().__init__(
            id=id,
            name=name,
            description=description,
            args_schema=BrowserToolForUrlSchema,
            url=url,  # type: ignore reportCallIssue
            model=model,
            infrastructure_option=infrastructure_option,
        )

    def run(  # type: ignore reportIncompatibleMethodOverride
        self,
        ctx: ToolRunContext,
        task: str,
        task_data: list[Any] | str | None = None,
    ) -> str | BaseModel | ActionClarification:
        """Run the BrowserToolForUrl."""
        return super().run(ctx, self.url, task, task_data)  # pragma: no cover


class BrowserInfrastructureProvider(ABC):
    """Abstract base class for browser infrastructure providers."""

    @abstractmethod
    def setup_browser(self, ctx: ToolRunContext) -> Browser:
        """Get a Browser instance.

        This is called at the start of every step using this tool.
        """

    @abstractmethod
    def construct_auth_clarification_url(self, ctx: ToolRunContext, sign_in_url: str) -> HttpUrl:
        """Construct the URL for the auth clarification."""

    @abstractmethod
    def step_complete(self, ctx: ToolRunContext) -> None:
        """Call when the step is complete to e.g. release the session if needed."""


class BrowserInfrastructureProviderLocal(BrowserInfrastructureProvider):
    """Browser infrastructure provider for local browser instances."""

    def __init__(
        self,
        chrome_path: str | None = None,
        extra_chromium_args: list[str] | None = None,
    ) -> None:
        """Initialize the BrowserInfrastructureProviderLocal."""
        self.chrome_path = chrome_path or self.get_chrome_instance_path()
        self.extra_chromium_args = extra_chromium_args or self.get_extra_chromium_args()

    def setup_browser(self, ctx: ToolRunContext) -> Browser:
        """Get a Browser instance.

        Note: This provider does not support end_user_id.

        Args:
            ctx (ToolRunContext): The context for the tool run, containing execution context
                and other relevant information.

        Returns:
            Browser: A configured Browser instance for local browser automation.

        """
        if ctx.end_user.external_id:
            logger().warning(
                "BrowserTool is using a local browser instance and does not support "
                "end users and so will be ignored.",
            )
        return Browser(
            config=BrowserConfig(
                chrome_instance_path=self.chrome_path,
                extra_chromium_args=self.extra_chromium_args or [],
            ),
        )

    def construct_auth_clarification_url(
        self,
        ctx: ToolRunContext,  # noqa: ARG002
        sign_in_url: str,
    ) -> HttpUrl:
        """Construct the URL for the auth clarification.

        Args:
            ctx (ToolRunContext): The context for the tool run, containing execution context
                and other relevant information.
            sign_in_url (str): The URL that the user needs to sign in to.

        Returns:
            HttpUrl: The URL for the auth clarification, which in this case is simply the sign-in
                URL passed directly through.

        """
        return HttpUrl(sign_in_url)

    def get_chrome_instance_path(self) -> str:
        """Get the path to the Chrome instance based on the operating system or env variable.

        Returns:
            str: The path to the Chrome executable. First checks for the
                PORTIA_BROWSER_LOCAL_CHROME_EXEC environment variable, then falls back to default
                locations based on the operating system.

        Raises:
            RuntimeError: If the platform is not supported (not macOS, Windows, or Linux) and the
            env variable isn't set.

        """
        chrome_path_from_env = os.environ.get("PORTIA_BROWSER_LOCAL_CHROME_EXEC")
        if chrome_path_from_env:
            return chrome_path_from_env

        match sys.platform:
            case "darwin":  # macOS
                return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            case "win32":  # Windows
                return r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
            case "linux":  # Linux
                return "/usr/bin/google-chrome"
            case _:
                raise RuntimeError(f"Unsupported platform: {sys.platform}")

    def step_complete(self, ctx: ToolRunContext) -> None:
        """Call when the step is complete to e.g release the session."""

    def get_extra_chromium_args(self) -> list[str] | None:
        """Get the extra Chromium arguments.

        Returns:
            list[str] | None: A list of extra Chromium arguments if the environment variable
                is set, otherwise None.

        """
        extra_chromium_args_from_env = os.environ.get("PORTIA_BROWSER_LOCAL_EXTRA_CHROMIUM_ARGS")
        if extra_chromium_args_from_env:
            return extra_chromium_args_from_env.split(",")
        return None


if BROWSERBASE_AVAILABLE:
    from browserbase import Browserbase

    class BrowserInfrastructureProviderBrowserBase(BrowserInfrastructureProvider):
        """Browser infrastructure provider for BrowserBase.

        This provider implements browser automation using BrowserBase's cloud infrastructure.
        It manages browser sessions and contexts for remote browser automation, with support
        for user-specific contexts.

        The provider requires both a BrowserBase API key and project ID, which can be provided
        either directly through the constructor or via environment variables (BROWSERBASE_API_KEY
        and BROWSERBASE_PROJECT_ID).

        Args:
            api_key (str, optional): The BrowserBase API key. If not provided, will be read from
                the BROWSERBASE_API_KEY environment variable.
            project_id (str, optional): The BrowserBase project ID. If not provided, will be read
                from the BROWSERBASE_PROJECT_ID environment variable.

        Raises:
            ToolHardError: If either the API key or project ID is not provided and cannot be found
                in environment variables.

        """

        def __init__(self, api_key: str | None = None, project_id: str | None = None) -> None:
            """Initialize the BrowserBase infrastructure provider.

            Args:
                api_key (str, optional): The BrowserBase API key. If not provided, will be read from
                    the BROWSERBASE_API_KEY environment variable.
                project_id (str, optional): The BrowserBase project ID. If not provided, will be
                    read from the BROWSERBASE_PROJECT_ID environment variable.

            Raises:
                ToolHardError: If either the API key or project ID is not provided and cannot be
                    found in environment variables.

            """
            api_key = api_key or os.environ.get("BROWSERBASE_API_KEY")
            if not api_key:
                raise ToolHardError("BROWSERBASE_API_KEY is not set")

            self.project_id = project_id or os.environ.get("BROWSERBASE_PROJECT_ID")
            if not self.project_id:
                raise ToolHardError("BROWSERBASE_PROJECT_ID is not set")

            self.bb = Browserbase(api_key=api_key)

        def step_complete(self, ctx: ToolRunContext) -> None:
            """Call when the step is complete closes the session to persist context."""
            # Only clean up the session on the final browser tool call
            if any(
                step.tool_id == "browser_tool"
                for step in ctx.plan.steps[ctx.plan_run.current_step_index + 1 :]
            ):
                return

            session_id = ctx.end_user.get_additional_data("bb_session_id")
            logger().debug(f"Closing BrowserBase session with id: {session_id}")
            if session_id:
                self.bb.sessions.update(
                    session_id,
                    project_id=self.project_id,  # type: ignore reportArgumentType
                    status="REQUEST_RELEASE",
                )
                ctx.end_user.set_additional_data("bb_session_id", "")
                ctx.end_user.set_additional_data("bb_session_connect_url", "")

        def get_context_id(self, ctx: ToolRunContext, bb: Browserbase) -> str:
            """Get the Browserbase context id.

            Creates a new context in the BrowserBase project. This method can be overridden in
            subclasses to implement custom context management, such as returning a saved context ID
            for a specific user.

            Args:
                ctx (ToolRunContext): The tool run context containing execution information.
                bb (Browserbase): The Browserbase client instance.

            Returns:
                str: The ID of the created or retrieved context.

            """
            if ctx.end_user.get_additional_data("bb_context_id"):
                logger().debug(
                    "Reusing existing context id: "
                    f"{ctx.end_user.get_additional_data('bb_context_id')}, "
                    f"end user: {ctx.end_user.external_id}"
                )
                return ctx.end_user.get_additional_data("bb_context_id")  # type: ignore reportReturnType

            fresh_context = bb.contexts.create(project_id=self.project_id)  # type: ignore reportArgumentType
            ctx.end_user.set_additional_data("bb_context_id", fresh_context.id)
            return fresh_context.id

        def create_session(self, bb_context_id: str) -> SessionCreateResponse:
            """Create a new BrowserBase session with the given context ID.

            Creates a persistent session that will remain active through clarification resolution.

            Args:
                bb_context_id (str): The BrowserBase context ID to associate with the session.

            Returns:
                SessionCreateResponse: The response containing session details including the
                    session ID and connection URL.

            """
            logger().debug(f"Creating BrowserBase session with context id: {bb_context_id}")
            return self.bb.sessions.create(
                project_id=self.project_id,  # type: ignore reportArgumentType
                browser_settings={
                    "context": {
                        "id": bb_context_id,
                        "persist": True,
                    },
                    "solve_captchas": False,  # We use HITL to solve captchas.
                },
                # keep_alive is needed so that the session can last through clarification
                # resolution.
                keep_alive=True,
            )

        def get_or_create_session(self, context: ToolRunContext, bb: Browserbase) -> str:
            """Get an existing session or create a new one if none exists.

            Manages session lifecycle by either retrieving an existing session from the context
            or creating a new one. Session details are stored in the execution context's
            additional_data for future retrieval.

            Args:
                context (ToolRunContext): The tool run context containing execution information.
                bb (Browserbase): The Browserbase client instance.

            Returns:
                str: The session connection URL that can be used to connect to the browser.

            """
            context_id = context.end_user.get_additional_data(
                "bb_context_id"
            ) or self.get_context_id(context, bb)

            context.end_user.set_additional_data("bb_context_id", context_id)

            session_id = context.end_user.get_additional_data("bb_session_id")
            session_connect_url = context.end_user.get_additional_data("bb_session_connect_url")

            if (
                # Force a new session for each plan run
                self._is_first_browser_tool_call(context.plan_run, context.plan)
                or not session_id
                or not session_connect_url
            ):
                session = self.create_session(context_id)
                session_connect_url = session.connect_url
                session_id = session.id
                context.end_user.set_additional_data("bb_session_id", session_id)
                context.end_user.set_additional_data("bb_session_connect_url", session_connect_url)
            else:
                logger().debug(f"Reusing existing BrowserBase session with id: {session_id}")

            logger().info(
                f"Browser tool debug link: {self.bb.sessions.debug(session_id).debugger_url}"
            )

            return session_connect_url

        def construct_auth_clarification_url(
            self,
            ctx: ToolRunContext,
            sign_in_url: str,  # noqa: ARG002
        ) -> HttpUrl:
            """Construct the URL for authentication clarification.

            Creates URL that allows viewing the browser session during authentication flows.

            Args:
                ctx (ToolRunContext): The tool run context containing execution information.
                sign_in_url (str): The URL where authentication should occur (not used in this
                    implementation as we return the debug view URL instead).

            Returns:
                HttpUrl: The URL for the debug view of the browser session.

            Raises:
                ToolHardError: If no session ID is found in the context.

            """
            session_id = ctx.end_user.get_additional_data("bb_session_id")
            if not session_id:
                raise ToolHardError("Session ID not found")
            live_view_link = self.bb.sessions.debug(session_id)
            return HttpUrl(live_view_link.pages[-1].debugger_fullscreen_url)

        def setup_browser(self, ctx: ToolRunContext) -> Browser:
            """Set up a Browser instance connected to BrowserBase.

            Creates or retrieves a BrowserBase session and configures a Browser instance
            to connect to it using the Chrome DevTools Protocol (CDP).

            Args:
                ctx (ToolRunContext): The tool run context containing execution information.

            Returns:
                Browser: A configured Browser instance connected to the BrowserBase session.

            """
            session_connect_url = self.get_or_create_session(ctx, self.bb)

            return Browser(
                config=BrowserConfig(
                    cdp_url=session_connect_url,
                ),
            )

        def _is_first_browser_tool_call(self, plan_run: PlanRun, plan: Plan) -> bool:
            """Check if the current call is the first browser call in the plan run.

            This will return false if we have resumed after a clarification.
            """
            if len(plan_run.get_clarifications_for_step()) > 0:
                return False

            return not any(
                step.tool_id == "browser_tool" for step in plan.steps[: plan_run.current_step_index]
            )
