"""Unit tests for the version utility module."""

from unittest.mock import mock_open, patch

from portia.version import get_version


def test_get_version_installed_package() -> None:
    """Test get_version function when package is installed."""
    with patch("portia.version.version") as mock_version:
        mock_version.return_value = "0.4.9"
        assert get_version() == "0.4.9"


def test_get_version_from_pyproject_toml() -> None:
    """Test get_version function when reading from pyproject.toml."""
    with (
        patch("portia.version.version", side_effect=Exception("Package not found")),
        patch("pathlib.Path.exists", return_value=True),
        patch(
            "pathlib.Path.open",
            mock_open(
                read_data=(
                    'name = "portia-sdk-python"\n'
                    'version = "0.4.9"\n'
                    'description = "Portia Labs Python SDK"'
                )
            ),
        ),
    ):
        assert get_version() == "0.4.9"


def test_get_version_from_pyproject_toml_with_quotes() -> None:
    """Test get_version function when reading from pyproject.toml with single quotes."""
    with (
        patch("portia.version.version", side_effect=Exception("Package not found")),
        patch("pathlib.Path.exists", return_value=True),
        patch(
            "pathlib.Path.open",
            mock_open(
                read_data=(
                    'name = "portia-sdk-python"\n'
                    "version = '0.4.9'\n"
                    'description = "Portia Labs Python SDK"'
                )
            ),
        ),
    ):
        assert get_version() == "0.4.9"


def test_get_version_from_pyproject_toml_no_version_line() -> None:
    """Test get_version function when pyproject.toml exists but has no version line."""
    with (
        patch("portia.version.version", side_effect=Exception("Package not found")),
        patch("pathlib.Path.exists", return_value=True),
        patch(
            "pathlib.Path.open",
            mock_open(
                read_data=('name = "portia-sdk-python"\ndescription = "Portia Labs Python SDK"')
            ),
        ),
    ):
        assert get_version() == "unknown"


def test_get_version_pyproject_toml_not_found() -> None:
    """Test get_version function when pyproject.toml doesn't exist."""
    with (
        patch("portia.version.version", side_effect=Exception("Package not found")),
        patch("pathlib.Path.exists", return_value=False),
    ):
        assert get_version() == "unknown"


def test_get_version_file_read_error() -> None:
    """Test get_version function when file reading fails."""
    with (
        patch("portia.version.version", side_effect=Exception("Package not found")),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.open", side_effect=Exception("File read error")),
    ):
        assert get_version() == "unknown"


def test_get_version_path_resolution() -> None:
    """Test get_version function path resolution for pyproject.toml."""
    with (
        patch("portia.version.version", side_effect=Exception("Package not found")),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.open", mock_open(read_data='version = "0.4.9"')),
    ):
        result = get_version()
        assert result == "0.4.9"
