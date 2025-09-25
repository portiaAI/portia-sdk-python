"""Logging utilities for Portia core functionality.

This module contains logging helpers that were moved from the main Portia class
to improve code organization and reusability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from portia.config import Config

from portia.config import GenerativeModelsConfig
from portia.logger import logger


def log_models(config: Config) -> None:
    """Log the models set in the configuration.

    This function logs all the generative models configured in the
    provided Config object for debugging purposes.

    Args:
        config: The Portia configuration containing model settings
    """
    logger().debug("Portia Generative Models")
    for model in GenerativeModelsConfig.model_fields:
        getter = getattr(config, f"get_{model}")
        logger().debug(f"{model}: {getter()}")