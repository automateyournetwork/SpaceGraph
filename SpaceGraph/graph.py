import os
import logging
import openai
from langgraph.graph import StateGraph, END
from state import State
from iss_locator import get_iss_location
from astronauts_in_space import get_astronauts
from langsmith.wrappers import wrap_openai
from langsmith import traceable

# ✅ Load Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO)

# ✅ Wrap OpenAI client with LangSmith tracing
client = wrap_openai(openai.Client())

# ✅ Initialize LangGraph StateGraph
graph = StateGraph(State)

# ✅ Define Available Tools
AVAILABLE_TOOLS = {
    "iss_agent": get_iss_location,  # ISS Locator
    "astronauts_agent": get_astronauts,  # Astronauts in Space Locator
}

# ✅ Assistant Node (Handles User Input and Routes to Correct Tool)
@traceable
def assistant_node(state: State) -> dict:
    user_input = state["user_input"]
    agent_responses = state.get("agent_response", [])

    # ✅ If responses exist, finalize NLP processing
    if agent_responses:
        logging.info(f"📝 Assistant finalizing response: {agent_responses}")

        # Extract ISS location and astronauts list
        iss_location = None
        astronauts_list = None

        for response in agent_responses:
            if isinstance(response, dict):  # ✅ Fix: Extract string from dict
                response = response.get("agent_response", "No response available.")
            
            if "Latitude" in response and "Longitude" in response:
                iss_location = response  # ✅ ISS Location response
            elif "astronauts in space" in response:
                astronauts_list = response  # ✅ Astronauts response

        # ✅ Merge ISS and astronaut data intelligently
        if iss_location and astronauts_list:
            merged_response = f"""
            {iss_location} 
            Based on current data, the following astronauts are in space: 
            {astronauts_list}
            """
            return {"final_response": merged_response, "next_node": "end"}

        # ✅ If only one tool provided data, return it as final response
        final_response = "\n".join(agent_responses)  # ✅ Fix: Strings only
        return {"final_response": final_response, "next_node": "end"}

    # Otherwise, classify and route question to necessary tools
    prompt = f"""
    You are an AI assistant that classifies user questions.
    Identify all tools needed to answer the question.

    Available tools: {', '.join(AVAILABLE_TOOLS.keys())}

    If multiple tools are needed, return them as a comma-separated list.
    If no tools apply, return "end".

    User Question: "{user_input}"

    Only return tool names from: {', '.join(AVAILABLE_TOOLS.keys())}, or "end".
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        categories = response.choices[0].message.content.strip().split(",")  
        categories = [c.strip() for c in categories]  
        logging.info(f"🔍 Assistant Routing Result: {categories}")
    except Exception as e:
        logging.error(f"⚠️ OpenAI API Error: {e}")
        categories = ["end"]

    valid_tools = [tool for tool in categories if tool in AVAILABLE_TOOLS]

    if not valid_tools:
        return {"next_node": "end"}

    return {"next_nodes": valid_tools}  # ✅ FIX: Use "next_nodes" for multiple calls

graph.add_node("assistant", assistant_node)

# ✅ ISS Locator Agent Node (Returns to Assistant)
@traceable
def iss_agent_node(state: State) -> dict:
    location_info = get_iss_location(state["user_input"])  # Fetch ISS Location

    # ✅ Append results as a string
    responses = state.get("agent_response", [])
    if isinstance(location_info, dict):  # ✅ Ensure correct format
        location_info = location_info.get("agent_response", "No response available.")
    responses.append(location_info)

    return {"agent_response": responses, "next_node": "assistant"}  # ✅ Return to Assistant

graph.add_node("iss_agent", iss_agent_node)

# ✅ Astronauts Locator Agent Node (Returns to Assistant)
@traceable
def astronauts_agent_node(state: State) -> dict:
    astronauts_info = get_astronauts(state["user_input"])  # Fetch Astronaut Data

    # ✅ Append results as a string
    responses = state.get("agent_response", [])
    if isinstance(astronauts_info, dict):  # ✅ Ensure correct format
        astronauts_info = astronauts_info.get("agent_response", "No response available.")
    responses.append(astronauts_info)

    return {"agent_response": responses, "next_node": "assistant"}  # ✅ Return to Assistant

graph.add_node("astronauts_agent", astronauts_agent_node)

# ✅ Set Entry Point
graph.set_entry_point("assistant")

# ✅ Routing Paths (Run multiple tools in parallel)
graph.add_conditional_edges(
    "assistant",
    lambda state: state.get("next_nodes", []),  # ✅ FIX: Properly check "next_nodes"
    {tool: tool for tool in AVAILABLE_TOOLS} | {"end": END}
)

# ✅ Ensure Tool Nodes Always Return to Assistant
graph.add_conditional_edges(
    "iss_agent",
    lambda state: state["next_node"],
    {
        "assistant": "assistant"
    }
)

graph.add_conditional_edges(
    "astronauts_agent",
    lambda state: state["next_node"],
    {
        "assistant": "assistant"
    }
)

# ✅ Compile the Graph
compiled_graph = graph.compile()
logging.info("🚀 LangGraph Compiled Successfully with Scalable Tools")
