import os
import time
import logging
import requests
from dotenv import load_dotenv
from langsmith import traceable

# ✅ Load Environment Variables
load_dotenv()

# ✅ Set API Key from .env (if needed)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO)

# ✅ ISS Location API URL
ISS_API_URL = "http://api.open-notify.org/iss-now.json"

class ISSLocator:
    def __init__(self):
        self.api_url = ISS_API_URL

    @traceable  # ✅ Enable LangSmith tracing for the ISS API call
    def get_location(self):
        """
        Fetches the ISS's current location.
        Handles retries and API failures gracefully.
        """
        retries = 3  # Retry up to 3 times in case of failure
        for attempt in range(retries):
            try:
                response = requests.get(self.api_url, timeout=5)  # Timeout for safety
                response.raise_for_status()
                data = response.json()

                # ✅ Extract ISS location
                latitude = data["iss_position"]["latitude"]
                longitude = data["iss_position"]["longitude"]

                return f"The ISS is currently at Latitude: {latitude}, Longitude: {longitude}."
            except requests.exceptions.RequestException as e:
                logging.error(f"❌ API Request Failed (Attempt {attempt+1}): {e}")
                time.sleep(2)  # Small delay before retrying
        return "⚠️ Unable to retrieve ISS location after multiple attempts."

# ✅ Define Function for LangGraph Node
@traceable  # ✅ Track ISS location retrieval
def get_iss_location(state):
    """
    Fetches the ISS's current location and returns the result as a structured response.
    """
    locator = ISSLocator()
    location_info = locator.get_location()
    return {"agent_response": location_info}  # ✅ Fix: Return a dictionary
