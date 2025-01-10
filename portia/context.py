"""Provides execution context to the planner and agents."""

from __future__ import annotations

from contextlib import contextmanager
from threading import local
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Generator

# Thread-local storage for end-user data
_execution_context = local()


class ExecutionContext(BaseModel):
    """Execution context provides runtime information to the runner, planner and agents.

    Unlike config it is designed to be used per request to allow customizing at that level.
    """

    end_user_id: str | None = None

    additional_data: dict[str, str] = {}

    # System Context Overrides
    # Generally be mindful of context window sizes when passing additional data through these field.

    # planner_system_context_extension allows passing additional context to the
    # planner LLMs. Useful for refining instructions or passing pointers.
    planner_system_context_extension: list[str] | None = None
    # agent_system_context_extension allows passing additional context to the
    # agent LLMs. Useful for passing execution hints or other data.
    agent_system_context_extension: list[str] | None = None


def empty_context() -> ExecutionContext:
    """Return an empty context."""
    return ExecutionContext(
        end_user_id=None,
        additional_data={},
        planner_system_context_extension=None,
        agent_system_context_extension=None,
    )


@contextmanager
def execution_context(
    context: ExecutionContext | None = None,
    end_user_id: str | None = None,
    metadata: dict[str, str] | None = None,
    planner_system_context_extension: list[str] | None = None,
    agent_system_context_extension: list[str] | None = None,
) -> Generator[None, None, None]:
    """Set the end-user context for the duration of the workflow."""
    if context is None:
        context = ExecutionContext(
            end_user_id=end_user_id,
            metadata=metadata or {},
            planner_system_context_extension=planner_system_context_extension,
            agent_system_context_extension=agent_system_context_extension,
        )
    _execution_context.context = context
    try:
        yield
    finally:
        _execution_context.context = None


def get_execution_context() -> ExecutionContext:
    """Retrieve the current end-user from the context."""
    return getattr(_execution_context, "context", empty_context())
