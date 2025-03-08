import json
import logging
from langgraph.graph import StateGraph
from typing import Dict, Any

# Agent imports
from state import State
from iss_locator_node import iss_locator_node
from astros_in_space_node import astros_in_space_node
from llm_router_node import llm_router_node
from weather_node import weather_node 

logging.basicConfig(level=logging.INFO)

# ✅ Initialize LangGraph StateGraph
graph = StateGraph(State)

# ✅ Add Nodes
graph.add_node("router_node", llm_router_node)
graph.add_node("iss_locator_node", iss_locator_node)
graph.add_node("astros_in_space_node", astros_in_space_node)
graph.add_node("weather_node", weather_node)

# ✅ Routing Function (Ensures Correct Execution)
def routing_function(state: Dict[str, Any]) -> str:
    """Determine the next step based on user intent."""
    next_step = state.get("next_agent", "__end__")  # ✅ Default to `__end__`
    
    logging.info(f"🚀 Routing Decision: {next_step}, Current State: {json.dumps(state, indent=2)}")

    # Ensure `next_step` is a valid node
    valid_steps = {"iss_locator_node", "astros_in_space_node", "weather_node", "__end__"}
    
    if next_step in valid_steps:
        return next_step  # ✅ Ensures valid return value

    # 🚨 Catch-all: If nothing valid is found, stop execution
    logging.warning(f"⚠️ No valid next step found: {next_step}. Stopping execution.")
    return "__end__"

# ✅ Weather Routing Function (Ensures ISS location is fetched first)
def weather_routing_function(state: Dict[str, Any]) -> str:
    """Ensure we fetch ISS location before getting weather, only if needed."""
    
    # ✅ Ensure `weather_node` is the next step when weather was requested
    if state.get("weather_requested", False):
        iss_location = state.get("iss_location", {})
        if not iss_location or "latitude" not in iss_location or "longitude" not in iss_location:
            logging.warning("⚠️ ISS location missing. Fetching ISS location first.")
            state["next_step"] = "iss_locator_node"  # ✅ Use next_step instead of return
            return state["next_step"]

        logging.info(f"✅ ISS location found: {iss_location}. Routing to weather_node.")
        state["next_step"] = "weather_node"  # ✅ Use next_step instead of return
        return state["next_step"]

    logging.info("✅ No weather request detected. Ending execution.")
    state["next_step"] = "__end__"
    return state["next_step"]


# ✅ Set Entry Point
graph.set_entry_point("router_node")

# ✅ Define Main Routing Paths
graph.add_conditional_edges(
    "router_node",
    routing_function,
    {
        "iss_locator_node": "iss_locator_node",
        "astros_in_space_node": "astros_in_space_node",
        "weather_node": "weather_node",
        "__end__": "__end__",  # ✅ Explicitly allow stopping
    }
)

# ✅ Ensure ISS locator correctly hands off to weather if needed
graph.add_conditional_edges(
    "iss_locator_node",
    weather_routing_function,  
    {
        "weather_node": "weather_node",
        "__end__": "__end__",  # ✅ Stop if weather isn't needed
    }
)

# ✅ Explicitly define `weather_node` as a valid transition and endpoint
graph.add_conditional_edges(
    "weather_node",
    lambda state: "__end__",  # ✅ Ensure execution always stops after weather_node
    {
        "__end__": "__end__",
    }
)

# ✅ Set Finish Points
graph.set_finish_point("iss_locator_node")
graph.set_finish_point("astros_in_space_node")
graph.set_finish_point("weather_node")

# ✅ Compile the Graph
compiled_graph = graph.compile()
