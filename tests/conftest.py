"""Configuration for pytest."""

import dotenv
import pytest


@pytest.fixture(scope="session", autouse=True)
def load_env_vars() -> None:
    """Load environment variables from .env file for testing."""
    dotenv.load_dotenv(override=True)
