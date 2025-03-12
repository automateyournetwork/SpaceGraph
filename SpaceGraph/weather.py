import os
import time
import logging
import requests
from dotenv import load_dotenv
from langsmith import traceable

# ✅ Load Environment Variables
load_dotenv()

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO)

# ✅ Weather API Configuration
BASE_API_URL = "https://api.weatherapi.com/v1"
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # Ensure this is set in your .env file

class WeatherFetcher:
    """Fetches weather information based on city or coordinates."""

    def __init__(self):
        self.api_url = f"{BASE_API_URL}/current.json"

    @traceable  # ✅ Enable LangSmith tracing
    def get_weather(self, city=None, lat=None, lon=None):
        """
        Retrieves the current weather for a given city or lat/lon.
        """
        if not WEATHER_API_KEY:
            logging.error("❌ WEATHER_API_KEY is missing. Check your .env file.")
            return "⚠️ Missing API Key for WeatherAPI."

        params = {"key": WEATHER_API_KEY}

        if city:
            params["q"] = city
        elif lat is not None and lon is not None:
            params["q"] = f"{lat},{lon}"
        else:
            logging.error("❌ Invalid parameters: Provide either a city or latitude/longitude.")
            return "⚠️ Please provide a valid city name or latitude/longitude."

        retries = 3  # Retry up to 3 times
        for attempt in range(retries):
            try:
                response = requests.get(self.api_url, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()

                # ✅ Extract relevant weather details
                location = data.get("location", {}).get("name", "Unknown Location")
                country = data.get("location", {}).get("country", "Unknown Country")
                temp_c = data.get("current", {}).get("temp_c", "N/A")
                condition = data.get("current", {}).get("condition", {}).get("text", "N/A")

                return f"The weather in {location}, {country} is {temp_c}°C with {condition}."

            except requests.exceptions.RequestException as e:
                logging.error(f"❌ Weather API Request Failed (Attempt {attempt+1}): {e}")
                time.sleep(2)  # Small delay before retrying

        return "⚠️ Unable to retrieve weather data after multiple attempts."

# ✅ Define Function for LangGraph Node
@traceable
def get_weather(params):
    """
    Fetches weather based on city or lat/lon and returns structured data.
    """
    city = params.get("city", None)
    lat = params.get("lat", None)
    lon = params.get("lon", None)

    # ✅ Log received parameters for debugging
    logging.info(f"🔍 Received weather parameters: City={city}, Lat={lat}, Lon={lon}")

    if not city and (not lat or not lon):
        logging.error("❌ Invalid parameters: Provide either a city or latitude/longitude.")
        return "⚠️ Please provide a valid city name or latitude/longitude."

    weather_fetcher = WeatherFetcher()
    weather_info = weather_fetcher.get_weather(city=city, lat=lat, lon=lon)

    return {"agent_response": weather_info}  # ✅ Structured response