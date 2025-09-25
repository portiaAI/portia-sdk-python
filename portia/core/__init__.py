"""Core infrastructure for Portia.

This module provides the foundational infrastructure for Portia, including
the PortiaContext class that holds immutable dependencies and lightweight
dataclasses for execution state management.
"""

from portia.core.context import PortiaContext
from portia.core.execution_state import PlanRunSession

__all__ = ["PortiaContext", "PlanRunSession"]