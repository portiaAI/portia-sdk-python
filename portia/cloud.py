"""Core client for interacting with portia cloud."""

import httpx

from portia.config import Config


class PortiaCloudClient:
    """Base HTTP client builder for interacting with portia cloud."""

    config: Config

    def __init__(self, config: Config) -> None:
        """Initialize the PortiaCloudClient instance.

        Args:
            config (Config): The Portia Configuration instance, containing the API key and endpoint.

        """
        self.config = config

    @classmethod
    def new_client(
        cls,
        config: Config,
        *,
        allow_unauthenticated: bool = False,
        json_headers: bool = True,
    ) -> httpx.Client:
        """Create a new httpx client.

        Args:
            config (Config): The Portia Configuration instance, containing the API key and endpoint.
            allow_unauthenticated (bool): Whether to allow creation of an unauthenticated client.
            json_headers (bool): Whether to add json headers to the request.

        """
        headers = {}
        if json_headers:
            headers = {
                "Content-Type": "application/json",
            }
        if config.portia_api_key or allow_unauthenticated is False:
            api_key = config.must_get_api_key("portia_api_key").get_secret_value()
            headers["Authorization"] = f"Api-Key {api_key}"
        return httpx.Client(
            base_url=config.must_get("portia_api_endpoint", str),
            headers=headers,
            timeout=httpx.Timeout(60),
            limits=httpx.Limits(max_connections=10),
        )

    def async_client(
        self,
        *,
        allow_unauthenticated: bool = False,
        json_headers: bool = True,
    ) -> httpx.AsyncClient:
        """Create a new httpx async client.

        Args:
            config (Config): The Portia Configuration instance, containing the API key and endpoint.
            allow_unauthenticated (bool): Whether to allow creation of an unauthenticated client.
            json_headers (bool): Whether to add json headers to the request.

        """
        headers = {}
        if json_headers:
            headers = {
                "Content-Type": "application/json",
            }
        if self.config.portia_api_key or allow_unauthenticated is False:
            api_key = self.config.must_get_api_key("portia_api_key").get_secret_value()
            headers["Authorization"] = f"Api-Key {api_key}"
        return httpx.AsyncClient(
            base_url=self.config.must_get("portia_api_endpoint", str),
            headers=headers,
            timeout=httpx.Timeout(60),
            limits=httpx.Limits(max_connections=10),
        )
