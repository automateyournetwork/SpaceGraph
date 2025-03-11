import json
import logging
import os
from langgraph.graph import StateGraph
from typing import Dict, Any

# LangSmith Imports
from langsmith import traceable
from langsmith.wrappers import wrap_openai

# Load environment variables for LangSmith
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

if LANGSMITH_TRACING:
    logging.info("✅ LangSmith Tracing Enabled")

# ✅ Initialize LangGraph StateGraph
from state import State
from iss_locator_node import iss_locator_node
from astros_in_space_node import astros_in_space_node
from llm_router_node import llm_router_node
from weather_node import weather_node
from mars_weather_node import mars_weather_node  # ✅ NEW: Import Mars Weather Node
from apod_node import apod_node
from neo_node import neo_node
from natural_language_answer_node import natural_language_answer_node

logging.basicConfig(level=logging.INFO)
graph = StateGraph(State)

# ✅ Add Nodes
graph.add_node("router_node", llm_router_node)
graph.add_node("iss_locator_node", iss_locator_node)
graph.add_node("astros_in_space_node", astros_in_space_node)
graph.add_node("weather_node", weather_node)
graph.add_node("mars_weather_node", mars_weather_node)  # ✅ NEW: Add Mars Weather Node
graph.add_node("apod_node", apod_node)
graph.add_node("neo_node", neo_node)
graph.add_node("natural_language_answer_node", natural_language_answer_node)

# ✅ Routing Function (Ensures Correct Execution)
@traceable  # 🚀 Auto-trace function execution
def routing_function(state: Dict[str, Any]) -> str:
    """Determine the next step based on user intent."""
    next_step = state.get("next_agent", "__end__")  # ✅ Default to `__end__`

    logging.info(f"🚀 Routing Decision: {next_step}, Current State: {json.dumps(state, indent=2)}")

    # Ensure `next_step` is a valid node
    valid_steps = {
        "iss_locator_node",
        "astros_in_space_node",s
        "weather_node",
        "mars_weather_node",
        "apod_node",
        "neo_node",
        "natural_language_answer_node",  # ✅ Ensure this can be reached
        "__end__"
    }

    if next_step in valid_steps:
        return next_step  # ✅ Ensures valid return value

    # 🚨 Catch-all: If nothing valid is found, stop execution
    logging.warning(f"⚠️ No valid next step found: {next_step}. Stopping execution.")
    return "__end__"

# ✅ Weather Routing Function (Handles ISS & Earth weather)
@traceable
def weather_routing_function(state: Dict[str, Any]) -> str:
    """Ensure ISS location is fetched before getting weather if needed."""
    
    # ✅ Ensure ISS location is retrieved before checking its weather
    if state.get("weather_requested", False):
        iss_location = state.get("iss_location", {})
        if not iss_location or "latitude" not in iss_location or "longitude" not in iss_location:
            logging.warning("⚠️ ISS location missing. Fetching ISS location first.")
            state["next_step"] = "iss_locator_node"  # ✅ Fetch ISS location first
            return state["next_step"]

        logging.info(f"✅ ISS location found: {iss_location}. Routing to weather_node.")
        state["next_step"] = "weather_node"  # ✅ Fetch weather after ISS location
        return state["next_step"]

    logging.info("✅ No ISS weather request detected. Routing to natural language output.")
    return "natural_language_answer_node"

# ✅ Earth-Mars Weather Comparison Function
@traceable
def earth_mars_comparison_routing(state: Dict[str, Any]) -> str:
    """If user wants both Earth and Mars weather, fetch Mars weather first, then Earth weather."""
    
    if state.get("compare_weather", False):
        logging.info("🌍🔄🪐 User requested Earth-Mars weather comparison. Fetching Earth weather after Mars.")
        state["next_step"] = "weather_node"  # ✅ Earth weather after Mars
        return state["next_step"]

    logging.info("✅ No comparison needed. Routing to NLP.")
    return "natural_language_answer_node"

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
        "mars_weather_node": "mars_weather_node",  # ✅ NEW: Route Mars Weather Queries
        "apod_node": "apod_node",
        "neo_node": "neo_node",
        "__end__": "__end__"
    }
)

# ✅ Ensure ISS locator correctly hands off to weather if needed
graph.add_conditional_edges(
    "iss_locator_node",
    weather_routing_function,  
    {
        "weather_node": "weather_node",
        "natural_language_answer_node": "natural_language_answer_node",
    }
)

graph.add_conditional_edges(
    "astros_in_space_node",
    lambda state: "natural_language_answer_node",
    {"natural_language_answer_node": "natural_language_answer_node"}
)

graph.add_conditional_edges(
    "weather_node",
    lambda state: "natural_language_answer_node",
    {"natural_language_answer_node": "natural_language_answer_node"}
)

# ✅ Ensure Mars weather can be compared to Earth
graph.add_conditional_edges(
    "mars_weather_node",
    lambda state: "weather_node" if state.get("compare_weather", False) else "natural_language_answer_node",  
    {
        "weather_node": "weather_node",  # ✅ Fetch Earth weather if needed
        "natural_language_answer_node": "natural_language_answer_node"  # ✅ Otherwise, return Mars weather immediately
    }
)

# ✅ Ensure APOD response is formatted before exiting
graph.add_conditional_edges(
    "apod_node",
    lambda state: "natural_language_answer_node",
    {"natural_language_answer_node": "natural_language_answer_node"}
)

# ✅ Ensure NEO response is formatted before exiting
graph.add_conditional_edges(
    "neo_node",
    lambda state: "natural_language_answer_node",
    {"natural_language_answer_node": "natural_language_answer_node"}
)

# ✅ Set final node to end execution
graph.add_conditional_edges(
    "natural_language_answer_node",
    lambda state: "__end__",
    {"__end__": "__end__"}
)

# ✅ Set Finish Points
graph.set_finish_point("natural_language_answer_node")

# ✅ Compile the Graph
compiled_graph = graph.compile()

logging.info("🚀 LangGraph Compiled Successfully")
