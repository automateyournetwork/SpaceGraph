import requests
import logging
import os
from typing import Dict, Any
from langsmith import traceable

ISS_API_URL = "http://api.open-notify.org/iss-now.json"

# ✅ Enable LangSmith Tracing if configured
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

if LANGSMITH_TRACING:
    logging.info("✅ LangSmith Tracing Enabled for `iss_locator_node`")

@traceable  # 🚀 Auto-trace this function
def iss_locator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch the current location of the ISS and update the state."""
    
    logging.info("🚀 Fetching ISS location...")

    try:
        response = requests.get(ISS_API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()

        # ✅ Ensure 'iss_position' exists before accessing it
        if "iss_position" in data and "latitude" in data["iss_position"] and "longitude" in data["iss_position"]:
            state["iss_location"] = {
                "latitude": data["iss_position"]["latitude"],
                "longitude": data["iss_position"]["longitude"]
            }
            logging.info(f"✅ ISS Location Retrieved: {state['iss_location']}")
        else:
            logging.error("❌ Unexpected API response format. 'iss_position' key missing.")
            state["iss_location"] = {"latitude": None, "longitude": None}

    except requests.RequestException as e:
        logging.error(f"❌ Error fetching ISS location: {e}", exc_info=True)
        state["iss_location"] = {"latitude": None, "longitude": None}

    # ✅ Routing logic: If weather was requested, route to `weather_node`
    if state.get("weather_requested", False):
        logging.info("🌦️ Weather was requested. Routing to weather_node.")
        state["next_agent"] = "weather_node"
        state["next_step"] = "weather_node"
        return state  

    # ✅ Otherwise, stop execution after retrieving ISS location
    logging.info("✅ User only wanted ISS location. Ending execution.")
    state["next_agent"] = "__end__"
    state["next_step"] = "__end__"
    
    return state
