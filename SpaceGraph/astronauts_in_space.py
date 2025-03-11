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

# ✅ Astronauts API URL
ASTROS_API_URL = "http://api.open-notify.org/astros.json"

class AstronautsInSpace:
    """Fetches the current astronauts in space."""

    def __init__(self):
        self.api_url = ASTROS_API_URL

    @traceable  # ✅ Enable LangSmith tracing
    def get_astronauts(self):
        """
        Retrieves a list of astronauts currently in space.
        Handles API retries and errors gracefully.
        """
        retries = 3  # Retry up to 3 times
        for attempt in range(retries):
            try:
                response = requests.get(self.api_url, timeout=5)  # Timeout for safety
                response.raise_for_status()
                data = response.json()

                # ✅ Extract astronaut names
                astronauts = [person["name"] for person in data.get("people", [])]
                if astronauts:
                    return f"There are currently {len(astronauts)} astronauts in space: {', '.join(astronauts)}."
                return "No astronauts found in space."

            except requests.exceptions.RequestException as e:
                logging.error(f"❌ API Request Failed (Attempt {attempt+1}): {e}")
                time.sleep(2)  # Small delay before retrying

        return "⚠️ Unable to retrieve astronaut data after multiple attempts."

# ✅ Define Function for LangGraph Node
@traceable  # ✅ Track astronaut retrieval
def get_astronauts(state):
    """
    Fetches astronauts in space and returns the result as a structured response.
    """
    astro_locator = AstronautsInSpace()
    astronauts_info = astro_locator.get_astronauts()
    return {"agent_response": astronauts_info}  # ✅ Return structured response
