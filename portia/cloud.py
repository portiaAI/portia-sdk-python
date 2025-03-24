"""Core client for interacting with portia cloud."""

import httpx

from portia.config import Config


class PortiaCloudClient:
    """Base HTTP client for interacting with portia cloud."""

    _client = None

    @classmethod
    def get_client(cls, config: Config, *, allow_unauthenticated: bool = False) -> httpx.Client:
        """Return the client using a singleton pattern to help manage limits across the SDK."""
        if cls._client is None:
            headers = {
                "Content-Type": "application/json",
            }
            if config.portia_api_key or allow_unauthenticated is False:
                api_key = config.must_get_api_key("portia_api_key").get_secret_value()
                headers["Authorization"] = f"Api-Key {api_key}"
            cls._client = httpx.Client(
                base_url=config.must_get("portia_api_endpoint", str),
                headers=headers,
                timeout=httpx.Timeout(60),
                limits=httpx.Limits(max_connections=10),
            )
        return cls._client
