import os
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

@traceable  # ✅ Track astronaut retrieval with LangSmith
def get_astronauts(messages):
    """
    Fetches a list of astronauts currently in space.

    Parameters:
    - messages (list): The message history (not directly used, but required by LangGraph).

    Returns:
    - dict: A structured response containing astronaut information.

    Example Output:
    ```
    {
        "agent_response": "🚀 There are currently 5 astronauts in space: Astronaut A, Astronaut B, ..."
    }
    ```
    """
    try:
        response = requests.get(ASTROS_API_URL, timeout=5)  # ✅ Timeout for safety
        response.raise_for_status()
        data = response.json()

        # ✅ Extract astronaut names
        astronauts = [person["name"] for person in data.get("people", [])]
        
        if astronauts:
            return {
                "agent_response": f"🚀 There are currently {len(astronauts)} astronauts in space: {', '.join(astronauts)}."
            }
        
        return {"agent_response": "🚀 There are no astronauts in space at this time."}

    except requests.exceptions.Timeout:
        logging.error("❌ Astronauts API Request Timed Out.")
        return {"agent_response": "⚠️ The request timed out. Please try again later."}

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ API Request Failed: {e}")
        return {"agent_response": "⚠️ Unable to retrieve astronaut data at this time."}
