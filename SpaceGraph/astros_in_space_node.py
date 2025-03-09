import requests
import logging
import os
from typing import Dict, Any
from langsmith import traceable

ASTROS_API_URL = "http://api.open-notify.org/astros.json"

# ✅ Enable LangSmith Tracing if configured
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

if LANGSMITH_TRACING:
    logging.info("✅ LangSmith Tracing Enabled for `astros_in_space_node`")

@traceable  # 🚀 Auto-trace this function
def astros_in_space_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch the current astronauts in space and update state."""
    
    logging.info("🚀 Fetching astronauts in space...")
    
    try:
        response = requests.get(ASTROS_API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()

        state['astronauts'] = [
            {"name": person["name"], "craft": person["craft"]}
            for person in data.get("people", [])
        ]

        logging.info(f"✅ Astronauts Retrieved: {state['astronauts']}")

    except requests.RequestException as e:
        logging.error(f"❌ Error fetching astronauts: {e}", exc_info=True)
        state['astronauts'] = []

    # ✅ Ensure execution stops after retrieving astronauts
    state["next_agent"] = "__end__"
    
    return state
