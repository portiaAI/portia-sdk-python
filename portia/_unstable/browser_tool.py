"""Browser tools.

This module contains tools that can be used to navigate to a URL, authenticate the user,
and complete tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any

from browser_use import Agent, Browser, BrowserConfig, Controller
from browserbase import Browserbase
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from pydantic_core import PydanticUndefined

from portia.clarification import ActionClarification
from portia.errors import ToolHardError
from portia.model import LangChainGenerativeModel  # noqa: TC001 - used in Pydantic Schema
from portia.tool import Tool, ToolRunContext

if TYPE_CHECKING:
    from browserbase.types import SessionCreateResponse

logger = logging.getLogger(__name__)

NotSet: Any = PydanticUndefined


class BrowserToolForUrlSchema(BaseModel):
    """Input schema for the BrowserToolForUrl."""

    task: str = Field(
        ...,
        description="The task to be completed by the Browser tool.",
    )


class BrowserToolSchema(BaseModel):
    """Input schema for the BrowserTool."""

    url: HttpUrl = Field(
        ...,
        description="The URL to navigate to.",
    )
    task: str = Field(
        ...,
        description="The task to be completed by the Browser tool.",
    )


class BrowserAuthOutput(BaseModel):
    """Output of the Browser tool's authentication check."""

    human_login_required: bool
    login_url: str | None = Field(
        default=None,
        description="The URL to navigate to for login if the user is not authenticated.",
    )
    user_login_guidance: str | None = Field(
        default=None,
        description="Guidance for the user to login if they are not authenticated.",
    )


class BrowserTaskOutput(BaseModel):
    """Output of the Browser tool's task."""

    task_output: str
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
    """Options for the browser infrastructure provider."""

    LOCAL = "local"
    BROWSERBASE = "browserbase"


class BaseBrowserTool(Tool[str]):
    """TODO: Document this."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(init_var=True, default="browser_tool")
    name: str = Field(init_var=True, default="Browser Tool")
    description: str = Field(
        init_var=True,
        default=(
            "General purpose browser tool. Can be used to navigate to a URL and "
            "complete tasks. Should only be used if the task requires a browser "
            "and you are sure of the URL."
        ),
    )
    args_schema: type[BaseModel] = Field(init_var=True, default=BrowserToolSchema)
    output_schema: tuple[str, str] = ("str", "The Browser tool's response to the user query.")

    model: LangChainGenerativeModel | None = Field(
        default=None,
        exclude=True,
        description="The model to use for the BrowserTool. If not provided, "
        "the model will be resolved from the config.",
    )

    infrastructure_option: BrowserInfrastructureOption = Field(
        default=BrowserInfrastructureOption.BROWSERBASE,
        description="The infrastructure provider to use for the browser tool.",
    )

    @cached_property
    def infrastructure_provider(self) -> BrowserInfrastructureProvider:
        """Get the infrastructure provider instance (cached)."""
        if self.infrastructure_option == BrowserInfrastructureOption.BROWSERBASE:
            return BrowserInfrastructureProviderBrowserBase()
        return BrowserInfrastructureProviderLocal()

    def run(self, ctx: ToolRunContext, url: str, task: str) -> str | ActionClarification:
        """Run the BrowserTool."""
        model = self.model or ctx.config.resolve_langchain_model()
        llm = model.to_langchain()

        async def run_browser_tasks() -> str | ActionClarification:
            # First auth check
            auth_agent = Agent(
                task=(
                    f"Go to {url}. If the user is not signed in, please go to the sign in page, "
                    "and indicate that human login is required by returning "
                    "human_login_required=True, and the url of the sign in page as well as "
                    "what the user should do to sign in. If the user is signed in, please "
                    "return human_login_required=False."
                ),
                llm=llm,
                browser=self.infrastructure_provider.setup_browser(ctx),
                controller=Controller(
                    output_model=BrowserAuthOutput,
                ),
            )
            result = await auth_agent.run()
            auth_result = BrowserAuthOutput.model_validate(json.loads(result.final_result()))  # type: ignore reportArgumentType
            if auth_result.human_login_required:
                if auth_result.user_login_guidance is None or auth_result.login_url is None:
                    raise ToolHardError(
                        "Expected user guidance and login URL if human login is required",
                    )
                return ActionClarification(
                    user_guidance=auth_result.user_login_guidance,
                    action_url=HttpUrl(
                        self.infrastructure_provider.construct_auth_clarification_url(
                            ctx,
                            auth_result.login_url,
                        ),
                    ),
                    plan_run_id=ctx.plan_run_id,
                )

            # Main task
            task_agent = Agent(
                task=task,
                llm=llm,
                browser=self.infrastructure_provider.setup_browser(ctx),
                controller=Controller(
                    output_model=BrowserTaskOutput,
                ),
            )
            result = await task_agent.run()
            task_result = BrowserTaskOutput.model_validate(json.loads(result.final_result()))  # type: ignore reportArgumentType
            if task_result.human_login_required:
                if task_result.user_login_guidance is None or task_result.login_url is None:
                    raise ToolHardError(
                        "Expected user guidance and login URL if human login is required",
                    )
                return ActionClarification(
                    user_guidance=task_result.user_login_guidance,
                    action_url=HttpUrl(
                        self.infrastructure_provider.construct_auth_clarification_url(
                            ctx,
                            task_result.login_url,
                        ),
                    ),
                    plan_run_id=ctx.plan_run_id,
                )
            return task_result.task_output

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(run_browser_tasks())


class BrowserTool(BaseBrowserTool):
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
            provider to use. Can be either "local" or "browserbase". Defaults to "browserbase".

    """

    def run(self, ctx: ToolRunContext, url: str, task: str) -> str | ActionClarification:
        """Run the BrowserTool."""
        return super().run(ctx, url, task)


class BrowserToolForUrl(BaseBrowserTool):
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

    """

    url: str = Field(
        ...,
        description="The URL to navigate to.",
    )

    def __init__(  # noqa: PLR0913
        self,
        url: str,
        id: str | None = None,  # noqa: A002
        name: str | None = None,
        description: str | None = None,
        model: LangChainGenerativeModel | None = NotSet,
        infrastructure_option: BrowserInfrastructureOption | None = NotSet,
    ) -> None:
        """Initialize the BrowserToolForUrl."""
        http_url = HttpUrl(url)
        if not http_url.host:
            raise ToolHardError("Invalid URL, host must be provided.")
        domain_parts = http_url.host.split(".")
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

    def run(self, ctx: ToolRunContext, task: str) -> str | ActionClarification:  # type: ignore reportIncompatibleMethodOverride
        """Run the BrowserToolForUrl."""
        return super().run(ctx, self.url, task)


class BrowserInfrastructureProvider(ABC):
    """Abstract base class for browser infrastructure providers."""

    @abstractmethod
    def setup_browser(self, ctx: ToolRunContext) -> Browser:
        """Get a Browser instance."""

    @abstractmethod
    def construct_auth_clarification_url(self, ctx: ToolRunContext, sign_in_url: str) -> HttpUrl:
        """Construct the URL for the auth clarification."""


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
        """Get a Browser instance."""
        if ctx.execution_context.end_user_id:
            logger.warning(
                "BrowserTool is using a local browser instance and does not support "
                "end_user_id. end_user_id will be ignored.",
            )
        return Browser(
            config=BrowserConfig(
                chrome_instance_path=self.chrome_path,
                extra_chromium_args=self.extra_chromium_args or [],
            ),
        )

    def construct_auth_clarification_url(self, ctx: ToolRunContext, sign_in_url: str) -> HttpUrl:  # noqa: ARG002
        """Construct the URL for the auth clarification."""
        return HttpUrl(sign_in_url)

    def get_chrome_instance_path(self) -> str:
        """Get the path to the Chrome instance based on the operating system or env variable."""
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

    def get_extra_chromium_args(self) -> list[str] | None:
        """Get the extra Chromium arguments."""
        extra_chromium_args_from_env = os.environ.get("PORTIA_BROWSER_LOCAL_EXTRA_CHROMIUM_ARGS")
        if extra_chromium_args_from_env:
            return extra_chromium_args_from_env.split(",")
        return None


class BrowserInfrastructureProviderBrowserBase(BrowserInfrastructureProvider):
    """Browser infrastructure provider for BrowserBase."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the BrowserBase infrastructure provider."""
        api_key = api_key or os.environ["BROWSERBASE_API_KEY"]
        if not api_key:
            raise ToolHardError("BROWSERBASE_API_KEY is not set")
        self.bb = Browserbase(api_key=api_key)

    def get_context_id(self, bb: Browserbase) -> str:
        """Get the Browserbase context id.

        This method can be overridden to return a saved context ID for a user.
        """
        return bb.contexts.create(project_id=os.environ["BROWSERBASE_PROJECT_ID"]).id

    def create_session(
        self,
        bb_context_id: str,
    ) -> SessionCreateResponse:
        """Get a fresh session with the given context ID."""
        return self.bb.sessions.create(
            project_id=os.environ["BROWSERBASE_PROJECT_ID"],
            browser_settings={
                "context": {
                    "id": bb_context_id,
                    "persist": True,
                },
            },
            # keep_alive is needed so that the session can last through clarification resolution.
            keep_alive=True,
        )

    def get_or_create_session(self, context: ToolRunContext, bb: Browserbase) -> str:
        """Get or create a Browserbase session."""
        context_id = context.execution_context.additional_data.get(
            "bb_context_id",
            self.get_context_id(bb),
        )
        context.execution_context.additional_data["bb_context_id"] = context_id

        session_id = context.execution_context.additional_data.get("bb_session_id", None)
        session_connect_url = context.execution_context.additional_data.get(
            "bb_session_connect_url",
            None,
        )

        if not session_id or not session_connect_url:
            session = self.create_session(context_id)
            session_connect_url = session.connect_url
            context.execution_context.additional_data["bb_session_id"] = session_id = session.id
            context.execution_context.additional_data["bb_session_connect_url"] = (
                session_connect_url
            )

        return session_connect_url

    def construct_auth_clarification_url(self, ctx: ToolRunContext, sign_in_url: str) -> HttpUrl:  # noqa: ARG002
        """Construct the URL for the auth clarification."""
        if not ctx.execution_context.additional_data["bb_session_id"]:
            raise ToolHardError("Session ID not found")
        live_view_link = self.bb.sessions.debug(
            ctx.execution_context.additional_data["bb_session_id"],
        )
        return HttpUrl(live_view_link.debugger_fullscreen_url)

    def setup_browser(self, ctx: ToolRunContext) -> Browser:
        """Get a Browser instance."""
        session_connect_url = self.get_or_create_session(ctx, self.bb)

        return Browser(
            config=BrowserConfig(
                cdp_url=session_connect_url,
            ),
        )
