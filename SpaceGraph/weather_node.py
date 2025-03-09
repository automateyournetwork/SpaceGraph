import os
import json
import logging
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langsmith import traceable

# Load environment variables
load_dotenv()

# Set API Key
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)

BASE_API_URL = "https://api.weatherapi.com/v1"

class WeatherAPI:
    """Handles fetching weather data from WeatherAPI, supporting both city and lat/long queries."""

    def __init__(self, api_key: Optional[str] = WEATHER_API_KEY):
        if not api_key:
            raise ValueError("❌ Missing API Key for Weather API.")
        self.api_key = api_key

    @traceable  # 🚀 Auto-trace this function
    def fetch_weather(self, location: str) -> Dict[str, Any]:
        """
        Fetch weather data using either:
        - A city name (e.g., "Toronto")
        - Latitude/Longitude (e.g., "43.7,-79.4")
        """
        url = f"{BASE_API_URL}/current.json"
        params = {"key": self.api_key, "q": location}

        try:
            logging.info(f"🌍 Fetching weather for: {location}")
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logging.error(f"❌ Weather API Error: {data['error']['message']}")
                return {"error": f"Weather data unavailable: {data['error']['message']}"}

            return data

        except requests.exceptions.RequestException as e:
            logging.error(f"⚠️ Weather API request failed: {e}", exc_info=True)
            return {"error": "Failed to retrieve weather data."}

@traceable  # 🚀 Auto-trace the weather node function
def weather_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handles weather queries dynamically for city names OR latitude/longitude."""
    logging.info("🌦️ Processing weather request...")

    # ✅ Extract user-specified location from state
    location = state.get("location")

    # 🚀 Fix: If `weather_requested=True` and no location is provided, use ISS lat/long
    if state.get("weather_requested", False) and not location:
        iss_location = state.get("iss_location", {})
        latitude = iss_location.get("latitude")
        longitude = iss_location.get("longitude")

        if latitude and longitude:
            location = f"{latitude},{longitude}"
            logging.info(f"🛰️ Using ISS location for weather query: {location}")
        else:
            logging.warning("⚠️ No valid ISS location found.")
            state["weather"] = {"error": "No valid location provided for weather request."}
            state["next_step"] = "__end__"
            return state

    # 🚀 Fix: Ensure `location` is a valid string before querying API
    if not location or not isinstance(location, str) or len(location.strip()) == 0:
        logging.warning("⚠️ No valid location provided.")
        state["weather"] = {"error": "No valid location provided for weather request."}
        state["next_step"] = "__end__"
        return state

    # ✅ Fetch weather data
    weather_client = WeatherAPI()
    weather_data = weather_client.fetch_weather(location.strip())

    # ✅ Store the response in state
    state["weather"] = weather_data
    logging.info(f"✅ Weather Data Retrieved: {json.dumps(weather_data, indent=2)}")

    state["next_step"] = "__end__"
    logging.info("✅ Weather Node Execution Complete.")

    return state
