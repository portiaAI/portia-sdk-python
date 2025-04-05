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
from typing import TYPE_CHECKING

from browser_use import Agent, Browser, BrowserConfig, Controller
from browserbase import Browserbase
from pydantic import BaseModel, Field, HttpUrl

from portia.clarification import ActionClarification
from portia.config import LLM_TOOL_MODEL_KEY
from portia.errors import ToolHardError
from portia.tool import Tool, ToolRunContext

if TYPE_CHECKING:
    from browserbase.types import SessionCreateResponse

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


class BrowserToolNonLocal(Tool[str]):
    """General purpose browser tool. Customizable to user requirements."""

    model_config = {
        "arbitrary_types_allowed": True
    }

    id: str = "browser_tool_non_local"
    name: str = "Browser Tool Non Local"
    description: str = (
        "General purpose browser tool. Can be used to navigate to a URL and "
        "complete tasks. Should only be used if the task requires a browser "
        "and you are sure of the URL."
    )
    args_schema: type[BaseModel] = BrowserToolSchema
    output_schema: tuple[str, str] = ("str", "The Browser tool's response to the user query.")

    @staticmethod
    def get_bb_instance() -> Browserbase:
        """Get a Browserbase instance."""
        return Browserbase(api_key=os.environ["BROWSERBASE_API_KEY"])

    bb: Browserbase = Field(default_factory=get_bb_instance)

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

    def get_or_create_session(self, context: ToolRunContext, bb: Browserbase) -> tuple[str, str]:
        """Get or create a Browserbase session."""
        context_id = context.execution_context.additional_data.get(
            "bb_context_id", self.get_context_id(bb)
        )
        context.execution_context.additional_data["bb_context_id"] = context_id

        session_id = context.execution_context.additional_data.get("bb_session_id", None)
        session_connect_url = context.execution_context.additional_data.get(
            "bb_session_connect_url", None
        )

        if not session_id or not session_connect_url:
            session = self.create_session(context_id)
            session_connect_url = session.connect_url
            context.execution_context.additional_data["bb_session_id"] = session_id = session.id
            context.execution_context.additional_data["bb_session_connect_url"] = (
                session_connect_url
            )

        return (session_id, session_connect_url)

    def construct_auth_clarification_url(self, ctx: ToolRunContext, sign_in_url: str) -> str:
        """Construct the URL for the auth clarification."""
        if not ctx.execution_context.additional_data["bb_session_id"]:
            raise ToolHardError("Session ID not found")
        live_view_link = self.bb.sessions.debug(ctx.execution_context.additional_data["bb_session_id"])
        return live_view_link.debugger_fullscreen_url

    def run(self, ctx: ToolRunContext, url: str, task: str) -> str | ActionClarification:
        """Run the BrowserTool."""
        model = ctx.config.resolve_langchain_model(LLM_TOOL_MODEL_KEY)
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
                browser=self._setup_browser(ctx),
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
                    action_url=HttpUrl(self.construct_auth_clarification_url(ctx, auth_result.login_url)),
                    plan_run_id=ctx.plan_run_id,
                )

            # Main task
            task_agent = Agent(
                task=task,
                llm=llm,
                browser=self._setup_browser(ctx),
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
                    action_url=HttpUrl(self.construct_auth_clarification_url(ctx, task_result.login_url)),
                    plan_run_id=ctx.plan_run_id,
                )
            return task_result.task_output

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(run_browser_tasks())

    def _setup_browser(self, ctx: ToolRunContext) -> Browser:
        """Get the browser instance to be used by the tool."""
        bb = self.get_bb_instance()
        session_id, session_connect_url = self.get_or_create_session(ctx, bb)

        return Browser(
            config=BrowserConfig(
                cdp_url=session_connect_url,
            ),
        )