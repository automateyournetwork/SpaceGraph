import os
import logging
import requests
import json  # ✅ Import json to parse OpenAI's tool call arguments
from dotenv import load_dotenv
from langsmith import traceable

# ✅ Load Environment Variables
load_dotenv()

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO)

# ✅ Weather API Configuration
BASE_API_URL = "https://api.weatherapi.com/v1"
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # ✅ Ensure this is set in your .env file
WEATHER_API_URL = f"{BASE_API_URL}/current.json"

@traceable  # ✅ Enable LangSmith tracing
def get_weather(params):
    """
    Fetches weather information based on a city or lat/lon coordinates.

    Parameters:
    - params (dict): The extracted parameters from OpenAI's tool call.

    Returns:
    - dict: A structured response containing weather details.
    """
    try:
        # ✅ Parse OpenAI's tool call arguments (ensure they're in dictionary format)
        if isinstance(params, str):  
            params = json.loads(params)  # OpenAI tool calls return JSON strings

        city = params.get("city", None)
        lat = params.get("lat", None)
        lon = params.get("lon", None)

        # ✅ Log received parameters for debugging
        logging.info(f"🔍 Received weather parameters: City={city}, Lat={lat}, Lon={lon}")

        if not WEATHER_API_KEY:
            logging.error("❌ WEATHER_API_KEY is missing. Check your .env file.")
            return {"agent_response": "⚠️ Missing API Key for WeatherAPI."}

        if not city and (lat is None or lon is None):
            logging.error("❌ Invalid parameters: Provide either a city or latitude/longitude.")
            return {"agent_response": "⚠️ Please provide a valid city name or latitude/longitude."}

        # ✅ Prepare request parameters
        request_params = {"key": WEATHER_API_KEY}
        if city:
            request_params["q"] = city
        else:
            request_params["q"] = f"{lat},{lon}"

        # ✅ Send request
        response = requests.get(WEATHER_API_URL, params=request_params, timeout=5)
        response.raise_for_status()
        data = response.json()

        # ✅ Extract relevant weather details
        location = data.get("location", {}).get("name", "Unknown Location")
        country = data.get("location", {}).get("country", "Unknown Country")
        temp_c = data.get("current", {}).get("temp_c", "N/A")
        condition = data.get("current", {}).get("condition", {}).get("text", "N/A")

        return {
            "agent_response": f"🌤️ The weather in {location}, {country} is {temp_c}°C with {condition}."
        }

    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON decoding error: {e}")
        return {"agent_response": "⚠️ Error decoding weather request parameters."}

    except requests.exceptions.Timeout:
        logging.error("❌ Weather API Request Timed Out.")
        return {"agent_response": "⚠️ The request timed out. Please try again later."}

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Weather API Request Failed: {e}")
        return {"agent_response": "⚠️ Unable to retrieve weather data at this time."}
