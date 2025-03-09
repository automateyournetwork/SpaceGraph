import os
import time
import logging
import requests
from datetime import datetime
from typing import Dict, Any
from langsmith import traceable

# ✅ NASA NEO API URL
NEO_API_URL = "https://api.nasa.gov/neo/rest/v1/feed"

# ✅ Enable LangSmith Tracing if configured
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

if LANGSMITH_TRACING:
    logging.info("✅ LangSmith Tracing Enabled for `neo_node`")

@traceable  # 🚀 Auto-trace this function
def neo_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch Near-Earth Object (NEO) data from NASA API and update state."""

    logging.info("🚀 Fetching Near-Earth Object (NEO) data...")

    retries = 3  # Number of retries
    NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")  # Use DEMO_KEY as fallback
    today_date = datetime.today().strftime("%Y-%m-%d")  # Get today's date

    for attempt in range(retries):
        try:
            response = requests.get(
                NEO_API_URL, 
                params={"start_date": today_date, "end_date": today_date, "api_key": NASA_API_KEY}, 
                timeout=5
            )
            response.raise_for_status()
            data = response.json()

            # ✅ Ensure we have asteroid data
            if "near_earth_objects" not in data:
                state["neo"] = {"error": "NASA NEO API did not return valid asteroid data."}
                break

            asteroid_list = data["near_earth_objects"].get(today_date, [])

            # ✅ Extract up to 3 asteroids for readability
            neo_summary = []
            for asteroid in asteroid_list[:3]:  # Limit to 3 closest asteroids
                neo_summary.append({
                    "name": asteroid.get("name", "Unknown"),
                    "diameter_km": asteroid["estimated_diameter"]["kilometers"]["estimated_diameter_max"],
                    "velocity_kph": asteroid["close_approach_data"][0]["relative_velocity"]["kilometers_per_hour"],
                    "miss_distance_km": asteroid["close_approach_data"][0]["miss_distance"]["kilometers"],
                    "hazardous": asteroid.get("is_potentially_hazardous_asteroid", False),
                })

            # ✅ Store NEO details in state
            state["neo"] = {
                "date": today_date,
                "asteroids": neo_summary
            }

            logging.info(f"✅ NASA NEO Data Retrieved: {state['neo']}")
            break  # ✅ Exit loop if successful

        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Error fetching NASA NEO data (Attempt {attempt+1}): {e}", exc_info=True)
            time.sleep(2)  # Wait before retrying

    # ✅ Default to empty response if all retries failed
    if "neo" not in state:
        state["neo"] = {"error": "Failed to fetch NASA NEO data after multiple attempts."}

    # ✅ Routing logic: If NEO data is retrieved, route to `natural_language_answer_node`
    state["next_agent"] = "natural_language_answer_node"

    return state
