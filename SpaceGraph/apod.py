import os
import logging
import requests
from dotenv import load_dotenv
from langsmith import traceable

# ✅ Load Environment Variables
load_dotenv()

# ✅ Load API Key
NASA_API_KEY = os.getenv("NASA_API_KEY")

# ✅ NASA APOD API URL
APOD_API_URL = "https://api.nasa.gov/planetary/apod"

@traceable
def get_apod(state):
    """
    Fetches the Astronomy Picture of the Day (APOD) from NASA's API.
    """
    params = {"api_key": NASA_API_KEY}
    try:
        response = requests.get(APOD_API_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        # ✅ Extract Image & Explanation
        image_url = data.get("url", "No image available")
        explanation = data.get("explanation", "No explanation available")

        logging.info("✅ Successfully retrieved NASA APOD.")
        return {
            "agent_response": f"🪐 Astronomy Picture of the Day:\n{image_url}\n📖 Explanation:\n{explanation}"
        }

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ APOD Request Failed: {e}")
        return {"agent_response": "⚠️ Unable to retrieve Astronomy Picture of the Day at this time."}