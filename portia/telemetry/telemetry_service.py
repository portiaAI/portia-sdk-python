"""Telemetry service for capturing anonymized usage data."""

from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

from portia.telemetry.views import BaseTelemetryEvent

T = TypeVar("T")


def singleton(cls: type[T]) -> Callable[..., T]:
    """Create a decorator for a singleton instance of a class.

    Args:
        cls: The class to make a singleton.

    Returns:
        A wrapper function that ensures only one instance of the class exists.

    """
    instance: list[T | None] = [None]

    def wrapper(*args: Any, **kwargs: Any) -> T:
        if instance[0] is None:
            instance[0] = cls(*args, **kwargs)
        return instance[0]

    def reset() -> None:
        instance[0] = None

    wrapper.reset = reset  # type: ignore reportFunctionMemberAccess
    return wrapper


class BaseProductTelemetry(ABC):
    """Base interface for capturing anonymized telemetry data.

    This class provides the interface for telemetry collection.
    Telemetry has been disabled and all calls are no-op.

    """

    @abstractmethod
    def capture(self, event: BaseTelemetryEvent) -> None:
        """Capture and send a telemetry event.

        Args:
            event (BaseTelemetryEvent): The telemetry event to capture

        """


@singleton
class ProductTelemetry(BaseProductTelemetry):
    """No-op telemetry service.

    This class provides a minimal no-op implementation of telemetry collection.
    All telemetry calls are ignored and have no effect.

    """

    def __init__(self) -> None:
        """Initialize the no-op telemetry service."""
        pass

    def capture(self, event: BaseTelemetryEvent) -> None:
        """No-op capture method.

        Args:
            event (BaseTelemetryEvent): The telemetry event to capture (ignored)

        """
        pass
