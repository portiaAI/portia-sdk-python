"""Tool to get the weather from openweathermap."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field

from portia.errors import ToolHardError, ToolSoftError
from portia.tool import Tool

if TYPE_CHECKING:
    from portia.context import ExecutionContext


class WeatherToolSchema(BaseModel):
    """Input for WeatherTool."""

    city: str = Field(..., description="The city to get the weather for")


class WeatherTool(Tool[str]):
    """Get the weather for a given city."""

    id: str = "weather_tool"
    name: str = "Weather Tool"
    description: str = "Get the weather for a given city"
    args_schema: type[BaseModel] = WeatherToolSchema
    output_schema: tuple[str, str] = ("str", "String output of the weather with temp and city")

    def run(self, _: ExecutionContext, city: str) -> str:
        """Run the WeatherTool."""
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not api_key or api_key == "":
            raise ToolHardError("OPENWEATHERMAP_API_KEY is required")
        url = (
            f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        )
        response = httpx.get(url)
        response.raise_for_status()
        data = response.json()
        if "weather" not in data:
            raise ToolSoftError(f"No data found for: {city}")
        weather = data["weather"][0]["description"]
        if "main" not in data:
            raise ToolSoftError(f"No main data found for city: {city}")
        temp = data["main"]["temp"]
        return f"The current weather in {city} is {weather} with a temperature of {temp}°C."
