import os
import logging
import requests
from dotenv import load_dotenv
from langsmith import traceable

# ✅ Load Environment Variables
load_dotenv()

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO)

# ✅ ISS Location API URL
ISS_API_URL = "http://api.open-notify.org/iss-now.json"

@traceable  # ✅ Enable LangSmith tracing for the ISS API call
def get_iss_location(messages):
    """
    Fetches the current location of the International Space Station (ISS).

    Parameters:
    - messages (list): The message history (not directly used, but required by LangGraph).

    Returns:
    - dict: A structured response containing the ISS's latitude and longitude.

    Example Output:
    ```
    {
        "agent_response": "🛰️ The ISS is currently at Latitude: 48.1234, Longitude: -122.9876."
    }
    ```
    """
    try:
        response = requests.get(ISS_API_URL, timeout=5)  # ✅ Timeout for safety
        response.raise_for_status()
        data = response.json()

        # ✅ Extract ISS location
        latitude = data["iss_position"]["latitude"]
        longitude = data["iss_position"]["longitude"]

        return {
            "agent_response": f"🛰️ The ISS is currently at Latitude: {latitude}, Longitude: {longitude}."
        }

    except requests.exceptions.Timeout:
        logging.error("❌ ISS Location API Request Timed Out.")
        return {"agent_response": "⚠️ The request timed out. Please try again later."}

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ API Request Failed: {e}")
        return {"agent_response": "⚠️ Unable to retrieve ISS location at this time."}
