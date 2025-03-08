import os
import json
import time
import logging
import requests
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API Keys from .env
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)

BASE_API_URL = "https://api.weatherapi.com/v1"

# ---------------------------- WEATHER API AGENT ---------------------------- #
class WeatherAPI:
    def __init__(self):
        self.api_key = WEATHER_API_KEY

    def fetch_data(self, endpoint, params):
        url = f"{BASE_API_URL}/{endpoint}.json"
        params["key"] = self.api_key
        
        retries = 3
        for attempt in range(retries):
            try:
                logging.debug(f"Fetching data from {url} with params {params}")
                response = requests.get(url, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()

                # ✅ Check if API returned an error message
                if "error" in data:
                    logging.error(f"❌ Weather API Error: {data['error']['message']}")
                    return {"error": "Unable to gather weather data."}

                return data  # ✅ Successful response

            except requests.exceptions.RequestException as e:
                logging.error(f"⚠️ API request failed (Attempt {attempt+1}): {e}")
                time.sleep(2)  # Retry delay
        
        # ✅ If all retries fail, return a user-friendly error message
        return {"error": "Unable to gather weather data."}

    def get_weather(self, latitude, longitude):
        return self.fetch_data("current", {"q": f"{latitude},{longitude}"})

def weather_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch weather based on ISS coordinates or user input and return a clean error message if needed."""
    logging.info("🌍 Fetching weather data...")

    # ✅ Handle user-provided coordinates (e.g., "how is the weather at 10,10")
    location = state.get("user_input", "").strip()
    if "weather" in location.lower() and "," in location:
        coords = location.split("weather at")[-1].strip()
    else:
        coords = None

    if coords:
        try:
            latitude, longitude = map(str.strip, coords.split(","))
        except ValueError:
            latitude, longitude = None, None
    else:
        latitude = state.get("iss_location", {}).get("latitude")
        longitude = state.get("iss_location", {}).get("longitude")

    if not latitude or not longitude:
        logging.warning("⚠️ Missing location. Cannot fetch weather.")
        state["weather"] = {"error": "Unable to gather weather data."}
        state["next_step"] = "__end__"
        return state

    # ✅ Fetch weather data
    weather_client = WeatherAPI()
    weather_data = weather_client.get_weather(latitude, longitude)

    # ✅ Store weather or error message in state
    state["weather"] = weather_data
    logging.info(f"🌦️ Weather API Response: {json.dumps(weather_data, indent=2)}")

    state["next_step"] = "__end__"  # ✅ Explicitly ensure execution ends here
    logging.info("✅ Weather Node Complete. Ending Execution.")
    
    return state
