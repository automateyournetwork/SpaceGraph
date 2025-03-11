import os
import logging
import openai
from langgraph.graph import StateGraph, END
from state import State
from iss_locator import get_iss_location
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
}

# ✅ Assistant Node (Handles User Input and Responses)
@traceable
def assistant_node(state: State) -> dict:
    user_input = state["user_input"]
    agent_response = state.get("agent_response", None)

    # If agent_response exists, finalize response
    if agent_response:
        logging.info(f"📝 Assistant finalizing response with ISS data: {agent_response}")
        return {"final_response": agent_response, "next_node": "end"}

    # Otherwise, classify and route question
    prompt = f"""
    You are an AI assistant that classifies user questions.
    Identify which tool is best for answering the question.

    Available tools: {', '.join(AVAILABLE_TOOLS.keys())}

    If the user question matches a tool, return the tool name.
    Otherwise, return "end".

    User Question: "{user_input}"

    Only return one of: {', '.join(AVAILABLE_TOOLS.keys())}, or "end".
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        category = response.choices[0].message.content.strip()
        logging.info(f"🔍 Assistant Routing Result: {category}")
    except Exception as e:
        logging.error(f"⚠️ OpenAI API Error: {e}")
        category = "end"  # Default to exit if API fails

    return {
        "next_node": category if category in AVAILABLE_TOOLS else "end"
    }

graph.add_node("assistant", assistant_node)

# ✅ ISS Locator Agent Node (Always Returns to Assistant)
@traceable
def iss_agent_node(state: State) -> dict:
    location_info = get_iss_location(state["user_input"])  # Fetch ISS Location
    return {"agent_response": location_info, "next_node": "assistant"}  # ✅ Always return to Assistant

graph.add_node("iss_agent", iss_agent_node)

# ✅ Set Entry Point
graph.set_entry_point("assistant")

# ✅ Routing Paths
graph.add_conditional_edges(
    "assistant",
    lambda state: state["next_node"],
    {
        "iss_agent": "iss_agent",
        "end": END  # ✅ Correctly defines the finish point
    }
)

# ✅ Ensure ISS Agent Always Returns to Assistant
graph.add_conditional_edges(
    "iss_agent",
    lambda state: state["next_node"],
    {
        "assistant": "assistant"  # ✅ No direct exit
    }
)

# ✅ No Need for `set_finish_point(END)` Anymore

# ✅ Compile the Graph
compiled_graph = graph.compile()
logging.info("🚀 LangGraph Compiled Successfully with Assistant Handling Everything")
