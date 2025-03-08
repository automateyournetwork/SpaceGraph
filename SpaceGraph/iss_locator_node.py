import requests
import logging
from typing import Dict, Any

ISS_API_URL = "http://api.open-notify.org/iss-now.json"

def iss_locator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logging.info("🚀 Fetching ISS location...")

    try:
        response = requests.get(ISS_API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()

        # ✅ Store latitude and longitude in state
        state["iss_location"] = {
            "latitude": data["iss_position"]["latitude"],
            "longitude": data["iss_position"]["longitude"]
        }
        logging.info(f"✅ ISS Location Retrieved: {state['iss_location']}")

    except requests.RequestException as e:
        logging.error(f"❌ Error fetching ISS location: {e}")
        state["iss_location"] = {"latitude": None, "longitude": None}

    # ✅ Fix: Check if weather was originally requested
    if state.get("weather_requested", False):
        logging.info("🌦️ Weather was requested. Routing to weather_node.")
        state["next_agent"] = "weather_node"  # ✅ Ensure correct routing
        state["next_step"] = "weather_node"   # ✅ Explicitly set next step
        return state  

    logging.info("✅ User only wanted ISS location. Ending execution.")
    state["next_agent"] = "__end__"
    state["next_step"] = "__end__"  # ✅ Ensure execution stops if weather not needed
    return state  
