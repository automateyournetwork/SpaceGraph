import os
import json
import time
import logging
import requests
from typing import Dict, Any
from langsmith import traceable

# ✅ NASA Mars Weather API URL
MARS_WEATHER_API_URL = "https://api.nasa.gov/insight_weather/"

# ✅ Enable LangSmith Tracing if configured
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

if LANGSMITH_TRACING:
    logging.info("✅ LangSmith Tracing Enabled for `mars_weather_node`")

@traceable  # 🚀 Auto-trace this function
def mars_weather_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch the latest weather data from Mars and update the state."""

    logging.info("🚀 Fetching latest Mars weather data from NASA InSight API...")

    retries = 3  # Number of retries
    NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")  # Use DEMO_KEY as fallback
    params = {
        "api_key": NASA_API_KEY,
        "feedtype": "json",
        "ver": "1.0"
    }

    for attempt in range(retries):
        try:
            response = requests.get(MARS_WEATHER_API_URL, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            # ✅ Extract the latest Sol (Martian Day)
            sol_keys = data.get("sol_keys", [])
            if not sol_keys:
                state["mars_weather"] = {"error": "No valid Sol data available from NASA."}
                break

            latest_sol = sol_keys[-1]  # Get the most recent Sol
            sol_data = data.get(latest_sol, {})

            # ✅ Extract weather details, ensuring availability
            mars_weather = {
                "sol": latest_sol,
                "season": sol_data.get("Season", "Unknown"),
                "temperature_c": sol_data.get("AT", {}).get("av", "N/A"),
                "pressure_pa": sol_data.get("PRE", {}).get("av", "N/A"),
                "wind_speed_mps": sol_data.get("HWS", {}).get("av", "N/A"),
                "wind_direction": sol_data.get("WD", {}).get("most_common", {}).get("compass_point", "N/A"),
                "first_utc": sol_data.get("First_UTC", "N/A"),
                "last_utc": sol_data.get("Last_UTC", "N/A"),
            }

            # ✅ Store Mars Weather Data in State
            state["mars_weather"] = mars_weather

            logging.info(f"✅ Mars Weather Data Retrieved: {state['mars_weather']}")
            break  # ✅ Exit loop if successful

        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Error fetching Mars Weather Data (Attempt {attempt+1}): {e}", exc_info=True)
            time.sleep(2)  # Wait before retrying

    # ✅ Default to empty response if all retries failed
    if "mars_weather" not in state:
        state["mars_weather"] = {"error": "Failed to fetch Mars Weather data after multiple attempts."}
        logging.warning("⚠️ Mars Weather API failed after all retries.")

    # ✅ Ensure Mars Weather fetches Earth Weather if requested
    if state.get("compare_weather", False):
        logging.info("🌍🆚🪐 User requested Earth-Mars weather comparison. Fetching Earth weather next.")
        state["next_agent"] = "weather_node"
    else:
        state["next_agent"] = "natural_language_answer_node"

    # ✅ Ensure next step is explicitly set
    state["next_step"] = state["next_agent"]

    logging.info(f"🚀 Mars Weather Node State: {json.dumps(state, indent=2)}")
    
    return state
