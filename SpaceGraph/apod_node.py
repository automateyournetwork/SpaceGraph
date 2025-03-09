import os
import time
import logging
import requests
from typing import Dict, Any
from langsmith import traceable

# ✅ NASA APOD API URL
APOD_API_URL = "https://api.nasa.gov/planetary/apod"

# ✅ Enable LangSmith Tracing if configured
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

if LANGSMITH_TRACING:
    logging.info("✅ LangSmith Tracing Enabled for `apod_node`")

@traceable  # 🚀 Auto-trace this function
def apod_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch NASA's Astronomy Picture of the Day (APOD) and update state."""

    logging.info("🚀 Fetching NASA Astronomy Picture of the Day (APOD)...")

    retries = 3  # Number of retries
    NASA_API_KEY = os.getenv("NASA_API_KEY")

    for attempt in range(retries):
        try:
            response = requests.get(APOD_API_URL, params={"api_key": NASA_API_KEY}, timeout=5)
            response.raise_for_status()
            data = response.json()

            if "url" not in data:
                state["apod"] = {"error": "NASA APOD response is missing an image URL."}
                break

            # ✅ Store APOD details in state
            state["apod"] = {
                "title": data.get("title", "Unknown Title"),
                "description": data.get("explanation", "No description available."),
                "date": data.get("date", "Unknown Date"),
                "media_type": data.get("media_type", "unknown"),
                "url": data["url"],
            }

            logging.info(f"✅ NASA APOD Retrieved: {state['apod']}")
            break  # ✅ Exit loop if successful

        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Error fetching NASA APOD (Attempt {attempt+1}): {e}", exc_info=True)
            time.sleep(2)  # Wait before retrying

    # ✅ Default to empty response if all retries failed
    if "apod" not in state:
        state["apod"] = {"error": "Failed to fetch NASA APOD after multiple attempts."}

    # ✅ Routing logic: If APOD is retrieved, route to `natural_language_answer_node`
    state["next_agent"] = "natural_language_answer_node"

    return state
