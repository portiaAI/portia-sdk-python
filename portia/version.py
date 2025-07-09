"""Version utilities for Portia SDK."""

from importlib.metadata import version
from pathlib import Path


def get_version() -> str:
    """Get the current version of the Portia SDK.

    This function works both when the package is installed as a dependency
    and when run directly from source. When run from source, it attempts
    to read the version from pyproject.toml.

    Returns:
        str: The current version of the Portia SDK

    """
    try:
        # Try to get version from installed package metadata
        return version("portia-sdk-python")
    except Exception:  # noqa: BLE001
        # Fallback: try to read from pyproject.toml when running from source
        try:
            # Find the project root by looking for pyproject.toml
            current_dir = Path(__file__).parent.parent
            pyproject_path = current_dir / "pyproject.toml"

            if pyproject_path.exists():
                with pyproject_path.open() as f:
                    for line in f:
                        if line.strip().startswith("version = "):
                            # Extract version from "version = "0.4.9""
                            return line.split("=")[1].strip().strip("\"'")
            return "unknown"  # noqa: TRY300

        except Exception:  # noqa: BLE001
            return "unknown"
