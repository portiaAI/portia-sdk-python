"""Browserbase authenticated tools."""

from __future__ import annotations

import contextlib
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from browser_use import Browser, BrowserConfig
from browserbase import Browserbase
from browserbase.types import SessionCreateResponse
from playwright.sync_api import Page, sync_playwright
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from portia.clarification import ActionClarification, MultipleChoiceClarification
from portia.tool import Tool, ToolHardError, ToolRunContext

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

    # TODO(Emma): Should auth and run be defined here as an abstact method?


class BrowserUseBrowseBaseTool(BrowserBaseTool[BrowserUse]):
    """Browserbase tool that uses BrowserUse."""

    def setup_browser(self, cdp_url: str):
        browser = Browser(config=BrowserConfig(cdp_url=cdp_url))
        context = UseBrowserbaseContext(
            browser,
            BrowserContextConfig(
                wait_for_network_idle_page_load_time=10.0,
                highlight_elements=True,
            ),
        )

    def auth_and_run(self, context: ToolRunContext) -> None | ActionClarification:
        """Authenticate the user to the tool and run it."""


# SIGNED_IN: class profile-card
# STATE_2: SEMI SIGNED IN -- "Continue as" (maybe listen for popup?) --> assume yes
# STATE_1: Not signed in --> "Continue with Google" --> popup flow


class PlaywrightBrowserBaseTool(BrowserBaseTool[Page]):
    """Browserbase tool that uses Playwright."""

    def get_live_view_link(self, bb: Browserbase, session_id: str, _: ToolRunContext) -> str:
        """Get the live view link for the session."""
        live_view_links = bb.sessions.debug(session_id)
        return live_view_links.debugger_fullscreen_url

    def auth_and_run(self, context: ToolRunContext) -> None | ActionClarification:
        """Authenticate the user to the tool and run it.

        This method will return an ActionClarification if authentication is required, or the result
        of the tool if it is not.
        """
        bb = self.get_bb_instance()
        session_id, session_connect_url = self.get_or_create_session(context, bb)

        with sync_playwright() as playwright:
            chromium = playwright.chromium
            browser = chromium.connect_over_cdp(session_connect_url)
            browser_context = browser.contexts[0]
            page = browser_context.pages[0]

            if not self.check_auth(page, context):
                self.pre_auth(page, context)
                live_view_link = self.get_live_view_link(bb, session_id, context)
                return ActionClarification(
                    user_guidance="Authentication required. Please click link to continue.",
                    action_url=HttpUrl(live_view_link),
                    plan_run_id=context.plan_run_id,
                )
            # TODO HANDLE CLOSING THE SESSION

            return self.post_auth(page, context)


class LinkedInFindConnectionsToolSchema(BaseModel):
    """Schema for the LinkedInFindConnectionsTool."""

    connection_name: str = Field(
        ...,
        description="The name of the connection to find.",
    )


class LinkedInFindConnectionsTool(PlaywrightBrowserBaseTool, Tool[str]):
    """Tool to find connections on LinkedIn."""

    id: str = "linkedin_find_connections_tool"
    name: str = "LinkedIn Find Connections Tool"
    description: str = "Find connections on LinkedIn."
    args_schema: type[BaseModel] = LinkedInFindConnectionsToolSchema
    output_schema: tuple[str, str] = ("list[str]", "The name of the connections found on LinkedIn.")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the tool."""
        Tool.__init__(self, **kwargs)
        PlaywrightBrowserBaseTool.__init__(self, None)

    def check_auth(self, page: Page, _: ToolRunContext) -> bool:
        """Check if the user is authenticated."""
        page.goto("https://www.linkedin.com")
        with contextlib.suppress(Exception):
            # Check if there's a div with class profile-card
            return page.locator("div.profile-card").is_visible(timeout=3000)
        return False

    def get_live_view_link(self, bb: Browserbase, session_id: str, context: ToolRunContext) -> str:
        """Get the live view link for the session."""
        live_view_links = bb.sessions.debug(session_id)
        if (
            context.execution_context.additional_data.get("linked_in_auth_preference")
            == "Google sign-in"
        ):
            google_page = next(
                (page for page in live_view_links.pages if "Google" in page.title),
                None,
            )
            if google_page:
                return google_page.debugger_fullscreen_url
            raise ToolHardError("No Google sign-in page found.")
        return live_view_links.debugger_fullscreen_url

    def pre_auth(self, page: Page, context: ToolRunContext) -> None:
        """Pre-auth script to run before authentication."""
        # Accept cookies
        print("got to pre-auth")
        with contextlib.suppress(Exception):
            page.click("[data-control-name='ga-cookie.consent.accept.v4']", timeout=3000)

        # Handle Google sign-in
        if (
            context.execution_context.additional_data.get("linked_in_auth_preference")
            == "Google sign-in"
        ):
            with contextlib.suppress(Exception):
                google_button = page.locator('[aria-label*="google"]').first
                google_button.click()
                context.execution_context.additional_data["auth_state"] = (
                    "google_popup"  # MAKE NICER
                )
                return
            with contextlib.suppress(Exception):
                google_button = page.locator("div:has-text('Continue as')").first
                google_button.click()
                context.execution_context.additional_data["auth_state"] = (
                    "google_popup"  # MAKE NICER
                )
                return
        else:
            with contextlib.suppress(Exception):
                page.click("text=Sign in with email")
                return

        # Handle linkedIn web to app sign-in check. # TODO: MAKE NICER
        with contextlib.suppress(Exception):
            page.locator("div:has-text('Check your LinkedIn app')").first.click()
            context.execution_context.additional_data["auth_state"] = (
                "linkedin_app_popup"  # MAKE NICER
            )
            for _ in range(10):
                current_url = page.url
                if "checkpoint" not in current_url and self.check_auth(page, context):
                    return
                time.sleep(1)
            raise ToolHardError("Failed to sign in with LinkedIn app.")

    def post_auth(self, page: Page, context: ToolRunContext) -> list[str]:
        """Post-auth script to run after authentication."""
        # Find the profile views count based on the HTML structure
        print("got to post-auth")
        try:
            # Find the element containing "Profile viewers" text
            profile_viewers_element = page.locator("span:text('Profile viewers')").first

            # Navigate to the parent elements to find the count
            count_element = profile_viewers_element.locator(
                "xpath=../../../div[contains(@class, 'feed-identity-widget-item__icon-stat')]//strong",
            )

            # Get the text content and clean it
            profile_views_text = count_element.text_content()
            if profile_views_text:
                # Remove commas and convert to integer
                profile_views = int(profile_views_text.replace(",", ""))
                return [str(profile_views)]
            return ["No profile views count found"]
        except Exception as e:
            return [f"Error finding profile views: {e!s}"]

    def run(
        self,
        context: ToolRunContext,
        **kwargs: Any,
    ) -> str | ActionClarification | MultipleChoiceClarification:
        # This will return the first clarification that has the attribute 'argument_name' with value 'linked_in_auth_preference'
        # If no matching clarification is found, it returns None
        clarification_result = next(
            (
                c
                for c in context.clarifications
                if hasattr(c, "argument_name")
                and c.argument_name == "linked_in_auth_preference"
                and c.resolved
            ),
            None,
        )

        if clarification_result:
            context.execution_context.additional_data["linked_in_auth_preference"] = (
                clarification_result.response
            )
            return self.auth_and_run(context)

        if not context.execution_context.additional_data.get("linked_in_auth_preference"):
            return MultipleChoiceClarification(
                user_guidance="Please select your LinkedIn authentication method.",
                options=[
                    "Google sign-in",
                    "Email / password sign-in",
                ],
                plan_run_id=context.plan_run_id,
                argument_name="linked_in_auth_preference",
            )

        return MultipleChoiceClarification(
            user_guidance="Please select your LinkedIn authentication method.",
            options=[
                "Google sign-in",
                "Email / password sign-in",
            ],
            plan_run_id=context.plan_run_id,
            argument_name="linked_in_auth_preference",
        )


class EveryoneActiveToolSchema(BaseModel):
    """Schema for the EveryoneActiveTool."""


class EveryoneActiveTool(PlaywrightBrowserBaseTool, Tool[str]):
    """Tool to retrieve membership ID from Everyone Active."""

    id: str = "everyone_active_tool"
    name: str = "Everyone Active Tool"
    description: str = "Retrieves membership ID from Everyone Active. Takes no arguments."
    args_schema: type[BaseModel] = EveryoneActiveToolSchema
    output_schema: tuple[str, str] = ("str", "The membership ID of the user.")

    def check_auth(self, page) -> bool:
        """Check if the tool is authenticated."""
        page.goto("https://www.everyoneactive.com")
        with contextlib.suppress(Exception):
            page.click("#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll")

        page.click("#menu-item-231")
        # Wait for navigation and check the page title
        page.wait_for_load_state("networkidle")
        return page.title() == "Memberships - Everyone Active"

    def pre_auth(self, _):
        """Pre-auth script to run before authentication."""
        # We don't have to do anything here because the auth check did it already.

    def post_auth(self, page):
        """Post-auth script to run after authentication."""
        page.click("a[href='/my-details/']")
        page.wait_for_load_state("networkidle")

        # Get the membership ID using the text selector
        membership_id = (
            page.locator("text=Your Membership ID")
            .locator("xpath=following-sibling::p/span")
            .text_content()
        )

        return f"Membership ID: {membership_id}"

    def run(self, context: ToolRunContext) -> str | ActionClarification:
        # TODO(Emma): Need to close the session
        return self.auth_and_run(context)
