"""Weather for BT, using Open-Meteo (free, no API key required).

This is BT's only tool that needs internet access — everything else runs
entirely on the local machine. fetch_weather() is the shared core logic:
both GetWeatherTool (for voice/text "what's the weather" questions) and
main.py's sidebar widget call it, so the two never drift out of sync.

Two requests: geocode the city name to coordinates, then fetch current
weather for those coordinates — both Open-Meteo endpoints, no signup.
"""

from __future__ import annotations

import asyncio

import requests
from pydantic import BaseModel, Field

from bt_core.logging_setup import get_logger
from bt_core.tools.base import PermissionTier, Tool, ToolError

log = get_logger(__name__)

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
_REQUEST_TIMEOUT_S = 10

_WEATHER_CODE_DESCRIPTIONS = {
    0: "clear sky",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
}


class WeatherReport(BaseModel):
    """A resolved current-weather reading for one city."""

    city: str
    temperature_c: float
    description: str


async def fetch_weather(city: str) -> WeatherReport:
    """Look up current weather for a city.

    Args:
        city: The city name to search for.

    Returns:
        The current weather for the best-matching location.

    Raises:
        ToolError: If the city can't be found, or the weather service
            can't be reached.
    """
    try:
        latitude, longitude, resolved_name = await asyncio.to_thread(_geocode, city)
        temperature_c, weather_code = await asyncio.to_thread(_fetch_current, latitude, longitude)
    except ToolError:
        raise
    except requests.RequestException as exc:
        log.error("weather_request_failed", city=city, exc_info=True)
        raise ToolError("I couldn't reach the weather service right now.") from exc

    description = _WEATHER_CODE_DESCRIPTIONS.get(weather_code, "unknown conditions")
    return WeatherReport(city=resolved_name, temperature_c=temperature_c, description=description)


def _geocode(city: str) -> tuple[float, float, str]:
    """Resolve a city name to coordinates. Runs in a worker thread."""
    response = requests.get(_GEOCODE_URL, params={"name": city, "count": 1}, timeout=_REQUEST_TIMEOUT_S)
    response.raise_for_status()
    results = response.json().get("results")
    if not results:
        raise ToolError(f"I couldn't find a place called '{city}'.")
    match = results[0]
    return match["latitude"], match["longitude"], match["name"]


def _fetch_current(latitude: float, longitude: float) -> tuple[float, int]:
    """Fetch current temperature and weather code for coordinates. Runs in a worker thread."""
    response = requests.get(
        _WEATHER_URL,
        params={"latitude": latitude, "longitude": longitude, "current_weather": "true"},
        timeout=_REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    current = response.json()["current_weather"]
    return current["temperature"], current["weathercode"]


class GetWeatherArgs(BaseModel):
    """Arguments for get_weather."""

    city: str = Field(description="The city to get the weather for")


class GetWeatherTool(Tool):
    """Gets the current weather for a city."""

    name = "get_weather"
    description = "Get the current weather for a city"
    permission_tier = PermissionTier.SAFE

    def _args_model(self) -> type[BaseModel]:
        return GetWeatherArgs

    async def _run(self, args: GetWeatherArgs) -> str:
        report = await fetch_weather(args.city)
        return f"It's {report.temperature_c:.0f}°C and {report.description} in {report.city}."
