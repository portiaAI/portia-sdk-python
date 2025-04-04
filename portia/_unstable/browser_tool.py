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

from browser_use import Agent, Browser, BrowserConfig, Controller
from pydantic import BaseModel, Field, HttpUrl

from portia.clarification import ActionClarification
from portia.config import LLM_TOOL_MODEL_KEY
from portia.errors import ToolHardError
from portia.tool import Tool, ToolRunContext

logger = logging.getLogger(__name__)


class BrowserToolSchema(BaseModel):
    """Input for the BrowserTool."""

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


class BrowserTool(Tool[str]):
    """General purpose browser tool. Customizable to user requirements."""

    id: str = "browser_tool"
    name: str = "Browser Tool"
    description: str = (
        "General purpose browser tool. Can be used to navigate to a URL and "
        "complete tasks. Should only be used if the task requires a browser "
        "and you are sure of the URL."
    )
    args_schema: type[BaseModel] = BrowserToolSchema
    output_schema: tuple[str, str] = ("str", "The Browser tool's response to the user query.")

    @staticmethod
    def _get_chrome_instance_path() -> str:
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

    chrome_path: str = Field(default_factory=_get_chrome_instance_path)

    def run(self, ctx: ToolRunContext, url: str, task: str) -> str | ActionClarification:
        """Run the BrowserTool."""
        model = ctx.config.resolve_langchain_model(LLM_TOOL_MODEL_KEY)
        llm = model.to_langchain()

        if ctx.execution_context.end_user_id:
            logger.warning(
                "BrowserTool uses a local browser instance and does not support "
                "end_user_id. end_user_id will be ignored.",
            )

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
                browser=self._setup_browser(),
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
                    action_url=HttpUrl(auth_result.login_url),
                    plan_run_id=ctx.plan_run_id,
                )

            # Main task
            task_agent = Agent(
                task=task,
                llm=llm,
                browser=self._setup_browser(),
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
                    action_url=HttpUrl(task_result.login_url),
                    plan_run_id=ctx.plan_run_id,
                )
            return task_result.task_output

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(run_browser_tasks())

    def _setup_browser(self) -> Browser:
        """Get the browser instance to be used by the tool."""
        return Browser(
            config=BrowserConfig(
                chrome_instance_path=self.chrome_path,
            ),
        )


"""Browserbase authenticated tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from browserbase import Browserbase
from browserbase.types import SessionCreateResponse
from playwright.sync_api import Page
from pydantic import BaseModel, ConfigDict, Field

from portia.tool import Tool, ToolHardError

SUPPORTED_BASE_ELEMENTS = [
    Page,  # Playwright
    int,  # TODO(Emma): This actually isn't supported, but want to allow for it in the future.
]
T = TypeVar("T", *SUPPORTED_BASE_ELEMENTS)  # type: ignore reportGeneralTypeIssues


class BrowserbaseAuthenticator(ABC, Generic[T]):
    """Base class for Browserbase authenticators.

    Authenticators define page scripts to run to check if the user is authenticated, and to
    bring them to the sign-in page if they are not.
    """

    @abstractmethod
    def check_auth(self, _: T, context: ToolRunContext) -> bool:
        """Check if the tool is authenticated."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def pre_auth(self, _: T, context: ToolRunContext) -> None:
        """Pre-auth script to run before authentication."""
        raise NotImplementedError("Subclasses must implement this method.")


class BrowserBaseTool(ABC, Generic[T]):
    """Base class for Browserbase tools.

    This class provides a base implementation for browser-based tools that use Browserbase for
    authentication and session management. When fully implemented, it will produce action
    clarifications for session authentication like regular Portia cloud tools, which will enable
    the user to sign in, and the agent to proceed once they have confirmed that they have done so.

    In general, you should call `auth_and_run` at the start of the `run` method of your tool, and
    then implement `post_auth` under the assumption that the tool is authenticated.

    The tool requires a BROWSERBASE_PROJECT_ID environment variable to be set and a
    BROWSERBASE_API_KEY environment variable to be set which can be found on the
    [Browserbase dashboard](https://www.browserbase.com/settings). We also recommend that you change
    the default timeout as when a user handles an authentication clarification they must do so
    within the timeout.

    The tool is designed to be extended over time with different page crawling mechanisms, for
    the moment it only supports Playwright, but in the future we expect to add Stagehand and
    BrowserUse.

    If you want authentication to last across sessions, you should save the context ID against
    the end-user, and override the `get_context_id` method to return the saved context ID.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    authenticator: BrowserbaseAuthenticator[T] | None = None

    def __init__(self, authenticator: BrowserbaseAuthenticator[T] | None = None) -> None:
        """Initialize the tool.

        Args:
            authenticator (BrowserbaseAuthenticator[T] | None): The authenticator to use for the
                tool. If not provided, relevant auth methods must be implemented in the subclass.

        """
        self.authenticator = authenticator

    @abstractmethod
    def check_auth(self, script_object: T, context: ToolRunContext) -> bool:
        """Check if the tool is authenticated.

        This should run a browser based script to determine whether the user is already
        authenticated.
        """
        if self.authenticator:
            return self.authenticator.check_auth(script_object, context)
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def pre_auth(self, script_object: T, context: ToolRunContext) -> None:
        """Pre-auth script to run before authentication.

        This should run a browser based script to get the user to the sign-in page.
        """
        if self.authenticator:
            self.authenticator.pre_auth(script_object, context)
        else:
            raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def post_auth(self, _: T, __: ToolRunContext) -> Any:  # noqa: ANN401
        """Post-auth script to run after authentication.

        The browser based script to run once the user has authenticated.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_context_id(self, bb: Browserbase) -> str:
        """Get the Browserbase context id.

        This method can be overridden to return a saved context ID for a user.
        """
        return bb.contexts.create(project_id=os.environ["BROWSERBASE_PROJECT_ID"]).id

    def create_session(
        self,
        bb: Browserbase,
        bb_context_id: str,
    ) -> SessionCreateResponse:
        """Get a fresh session with the given context ID."""
        return bb.sessions.create(
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

    def get_bb_instance(self) -> Browserbase:
        """Get a Browserbase instance."""
        return Browserbase(api_key=os.environ["BROWSERBASE_API_KEY"])

    def get_or_create_session(self, context: ToolRunContext, bb: Browserbase) -> tuple[str, str]:
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
            session = self.create_session(bb, context_id)
            session_connect_url = session.connect_url
            context.execution_context.additional_data["bb_session_id"] = session_id = session.id
            context.execution_context.additional_data["bb_session_connect_url"] = (
                session_connect_url
            )

        return (session_id, session_connect_url)
