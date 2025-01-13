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
    """Execution context provides runtime information to the runner, planner, and agents.

    Unlike configuration settings, it is designed to be used on a per-request basis,
    allowing customization at runtime. For example, this can pass end-user-specific
    information to planners and agents for dynamic adjustments.

    ExecutionContext
    """

    # end_user_id is used by the system to identify the actual user for who a workflow is running.
    # This is used to identify the user for Authentication purposes and also to assist with
    # debugging and reporting. If provided the value should be a string that uniquely identifies
    # a user. For example it may be an internal user_id, email address or other unique attribute.
    end_user_id: str | None = None

    # additional_data allows passing additional data that may be useful to either the agents
    # executing the workflow or later for debugging/reporting. For example you may want to pass
    # the users email address here if using the email tool, or you may want to pass through an
    # internal tracing ID to make it easy to correlate logs in the Portia dashboard.
    additional_data: dict[str, str] = {}

    # planner_system_context_extension allows passing additional context to the
    # planner LLMs. Useful for refining instructions or passing hints. Note additional_data is also
    # passed to the agents so for ad-hoc data prefer additional_data.
    planner_system_context_extension: list[str] | None = None

    # agent_system_context_extension allows passing additional context to the
    # agent LLMs. Useful for passing execution hints. Note additional_data is also
    # passed to the agents so for ad-hoc data prefer additional_data.
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
    additional_data: dict[str, str] | None = None,
    planner_system_context_extension: list[str] | None = None,
    agent_system_context_extension: list[str] | None = None,
) -> Generator[None, None, None]:
    """Set the execution context for the current thread for the duration of the workflow.

    This context manager ensures thread safety by using thread-local storage,
    meaning that the execution context set within this block will only affect
    the current thread. This is particularly useful in multi-threaded
    applications, such as web servers or task queues, where multiple threads
    may need independent contexts simultaneously.

    Arguments:
    ---------
        context (Optional[ExecutionContext]): The execution context to set for this thread.
            If not provided, a new `ExecutionContext` is created using the provided parameters.
        end_user_id (Optional[str]): An identifier for the end user, used to customize
            the execution for specific users. Defaults to `None`.
        additional_data (Optional[Dict[str, str]]): Arbitrary additional data to associate
            with the context. Defaults to an empty dictionary.
        planner_system_context_extension (Optional[list[str]]): Additional context for planner
            LLMs. This should be concise to stay within the context window.
        agent_system_context_extension (Optional[list[str]]): Additional context for agent
            LLMs. This should also be concise.

    Yields:
    ------
        None: The block of code within the context manager executes with the specified context.

    Thread Safety:
        - The `_execution_context` object is a thread-local storage instance, ensuring that
          the `ExecutionContext` set in one thread does not affect others.
        - When the context manager exits, the context for the current thread is cleaned up
          to avoid memory leaks or unintended persistence of data.

    Example:
        >>> with execution_context(end_user_id="user123", additional_data={"key": "value"}):
        >>>     # Code here runs with the specified execution context
        >>> # Outside the block, the execution context is cleared for the current thread.

    """
    if context is None:
        context = ExecutionContext(
            end_user_id=end_user_id,
            additional_data=additional_data or {},
            planner_system_context_extension=planner_system_context_extension,
            agent_system_context_extension=agent_system_context_extension,
        )
    _execution_context.context = context
    try:
        yield
    finally:
        delattr(_execution_context, "context")


def get_execution_context() -> ExecutionContext:
    """Retrieve the current end-user from the context."""
    return getattr(_execution_context, "context", empty_context())
