import requests
import logging
from typing import Dict, Any

ASTROS_API_URL = "http://api.open-notify.org/astros.json"

def astros_in_space_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logging.info("🚀 Fetching astronauts in space...")

    try:
        response = requests.get(ASTROS_API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()

        state['astronauts'] = [
            {"name": person["name"], "craft": person["craft"]}
            for person in data["people"]
        ]
        logging.info(f"✅ Astronauts Retrieved: {state['astronauts']}")

    except requests.RequestException as e:
        logging.error(f"❌ Error fetching astronauts: {e}")
        state['astronauts'] = []

    # ✅ Ensure execution stops after retrieving astronauts
    state["next_agent"] = "__end__"
    return state
