"""Tests for runner classes."""

import tempfile
from pathlib import Path

import pytest
from pydantic import SecretStr

from portia.config import AgentType, Config, default_config
from portia.errors import ConfigNotFoundError, InvalidConfigError


def test_runner_config_from_file() -> None:
    """Test loading configuration from a file."""
    config_data = """{
"portia_api_key": "file-key",
"openai_api_key": "file-openai-key",
"llm_model_temperature": 10,
"storage_class": "MEMORY",
"llm_provider": "OPENAI",
"llm_model_name": "gpt-4o-mini",
"llm_model_seed": 443,
"default_agent_type": "VERIFIER"
}"""

    with tempfile.NamedTemporaryFile("w", delete=True, suffix=".json") as temp_file:
        temp_file.write(config_data)
        temp_file.flush()

        config_file = Path(temp_file.name)

        config = Config.from_file(config_file)

        assert config.must_get_raw_api_key("portia_api_key") == "file-key"
        assert config.must_get_raw_api_key("openai_api_key") == "file-openai-key"
        assert config.default_agent_type == AgentType.VERIFIER
        assert config.llm_model_temperature == 10


def test_getters() -> None:
    """Test getters work."""
    c = default_config()
    c.openai_api_key = SecretStr("123")

    assert c.has_api_key("portia_api_key")

    with pytest.raises(ConfigNotFoundError):
        c.must_get("not real", str)

    with pytest.raises(InvalidConfigError):
        c.must_get("portia_api_key", int)

    with pytest.raises(InvalidConfigError):
        c.must_get_raw_api_key("portia_api_key")

    with pytest.raises(InvalidConfigError):
        c.must_get_api_key("portia_api_key")
