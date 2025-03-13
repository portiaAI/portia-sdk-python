"""Core client for interacting with portia cloud."""

import httpx

from portia.config import Config


class PortiaCloudClient:
    """Base HTTP client for interacting with portia cloud."""

    _client = None

    @classmethod
    def get_client(cls, config: Config) -> httpx.Client:
        """Return the client using a singleton pattern to help manage limits across the SDK."""
        if cls._client is None:
            api_key = config.must_get_api_key("portia_api_key").get_secret_value()
            cls._client = httpx.Client(
                base_url=config.must_get("portia_api_endpoint", str),
                headers={
                    "Authorization": f"Api-Key {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60),
                limits=httpx.Limits(max_connections=10),
            )
        return cls._client
