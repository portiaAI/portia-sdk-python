"""Telemetry service for capturing anonymized usage data."""

import logging
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv
from posthog import Posthog

from portia.common import singleton
from portia.telemetry.views import BaseTelemetryEvent
from portia.version import get_version

load_dotenv(override=True)

logger = logging.getLogger(__name__)


def xdg_cache_home() -> Path:
    """Get the XDG cache home directory path.

    Returns:
        Path: The path to the cache directory, either from XDG_CACHE_HOME environment variable
              or the default ~/.portia location.

    """
    default = Path.home() / ".portia"
    env_var = os.getenv("XDG_CACHE_HOME")
    if env_var and (path := Path(env_var)).is_absolute():
        return path
    return default


def get_project_id_key() -> str:
    """Get the project ID key.

    Returns:
        str: The project ID key

    """
    if os.getenv("PORTIA_API_ENDPOINT"):
        endpoint = os.getenv("PORTIA_API_ENDPOINT")
        if "localhost" in endpoint:  # type: ignore reportOperatorIssue
            return "phc_QHjx4dKKNAqmLS1U64kIXo4NlYOGIFDgB1qYxw3wh1W"  # local / dev environment
        if "dev" in endpoint:  # type: ignore reportOperatorIssue
            return "phc_gkmBfAtjABu5dDAX9KX61iAF10Wyze4FGPrT3g7mcKo"  # staging environment
    return "phc_fGJERhs0sljicW5IFBzJZoenOb0jtsIcAghCZHw97V1"  # prod environment


class BaseProductTelemetry(ABC):
    """Base interface for capturing anonymized telemetry data.

    This class handles the collection and transmission of anonymized usage data to PostHog.
    Telemetry can be disabled by setting the environment variable `ANONYMIZED_TELEMETRY=False`.

    """

    @abstractmethod
    def capture(self, event: BaseTelemetryEvent) -> None:
        """Capture and send a telemetry event.

        Args:
            event (BaseTelemetryEvent): The telemetry event to capture

        """


@singleton
class ProductTelemetry(BaseProductTelemetry):
    """Service for capturing anonymized telemetry data.

    This class handles the collection and transmission of anonymized usage data to PostHog.
    Telemetry can be disabled by setting the environment variable `ANONYMIZED_TELEMETRY=False`.

    Attributes:
        USER_ID_PATH (str): Path where the user ID is stored
        PROJECT_API_KEY (str): PostHog project API key
        HOST (str): PostHog server host URL
        UNKNOWN_USER_ID (str): Default user ID when user identification fails

    """

    USER_ID_PATH = str(xdg_cache_home() / "portia" / "telemetry_user_id")
    PROJECT_API_KEY = get_project_id_key()
    HOST = "https://eu.i.posthog.com"
    UNKNOWN_USER_ID = "UNKNOWN"

    _curr_user_id = None

    def __init__(self) -> None:
        """Initialize the telemetry service.

        Sets up the PostHog client if telemetry is enabled and configures logging.
        """
        telemetry_disabled = os.getenv("ANONYMIZED_TELEMETRY", "true").lower() == "false"

        if telemetry_disabled:
            self._posthog_client = None
        else:
            logger.info(
                "Portia anonymized telemetry enabled. "
                "See https://docs.portialabs.ai/telemetry for more information."
            )
            self._posthog_client = Posthog(
                project_api_key=self.PROJECT_API_KEY,
                host=self.HOST,
                disable_geoip=False,
                enable_exception_autocapture=True,
            )
            self.debug_logging = logger.level == "DEBUG"

            if not self.debug_logging:
                # Silence posthog's logging
                posthog_logger = logging.getLogger("posthog")  # pragma: no cover
                posthog_logger.disabled = True  # pragma: no cover

        if self._posthog_client is None:
            logger.debug("Telemetry disabled")

    def capture(self, event: BaseTelemetryEvent) -> None:
        """Capture and send a telemetry event.

        Args:
            event (BaseTelemetryEvent): The telemetry event to capture

        """
        if self._posthog_client is None:
            return

        if self.debug_logging:
            logger.debug(f"Telemetry event: {event.name} {event.properties}")  # noqa: G004

        try:
            self._posthog_client.capture(
                event.name,
                distinct_id=self.user_id,
                properties={
                    **event.properties,
                    "process_person_profile": True,
                    "sdk_version": get_version(),
                },
            )
        except Exception:
            logger.exception(f"Failed to send telemetry event {event.name}")  # noqa: G004

    @property
    def user_id(self) -> str:
        """Get the current user ID, generating a new one if it doesn't exist.

        Returns:
            str: The user ID, either from cache or newly generated

        """
        if self._curr_user_id:
            return self._curr_user_id

        # File access may fail due to permissions or other reasons. We don't want to
        # crash so we catch all exceptions.
        try:
            if not Path(self.USER_ID_PATH).exists():
                Path(self.USER_ID_PATH).parent.mkdir(parents=True, exist_ok=True)
                with Path(self.USER_ID_PATH).open("w") as f:
                    new_user_id = str(uuid.uuid4())
                    f.write(new_user_id)
                self._curr_user_id = new_user_id
            else:
                with Path(self.USER_ID_PATH).open() as f:
                    self._curr_user_id = f.read()
        except Exception:  # noqa: BLE001
            self._curr_user_id = "UNKNOWN_USER_ID"
        return self._curr_user_id
