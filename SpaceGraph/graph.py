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
    # More tools can be added here dynamically
}

# ✅ Assistant Node (Routes to Tools or Exit)
@traceable
def assistant_node(state: State) -> dict:
    user_input = state["user_input"]

    prompt = f"""
    You are an AI assistant that classifies user questions.
    Identify which tool is best for answering the question.

    Available tools: {', '.join(AVAILABLE_TOOLS.keys())}

    If the user question matches a tool, return the tool name.
    If the question does not match any tool, return "exit_node".

    User Question: "{user_input}"

    Only return one of: {', '.join(AVAILABLE_TOOLS.keys())}, or "exit_node".
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        category = response.choices[0].message.content.strip()
        logging.info(f"🔍 ChatGPT Routing Result: {category}")
    except Exception as e:
        logging.error(f"⚠️ OpenAI API Error: {e}")
        category = "exit_node"  # Default to exit if API fails

    return {
        "next_node": "tools" if category in AVAILABLE_TOOLS else "exit_node",
        "selected_tool": category if category in AVAILABLE_TOOLS else None
    }

graph.add_node("assistant", assistant_node)

# ✅ Tools Node (Selects a Tool)
@traceable
def tools_node(state: State) -> dict:
    """
    Selects which tool to use.
    Ensures selected_tool is always present.
    """
    selected_tool = state.get("selected_tool", None)  # 🔥 Default to None

    if selected_tool in AVAILABLE_TOOLS:
        return {"next_tool": selected_tool}  # ✅ Valid tool selection
    else:
        return {"next_tool": "assistant"}  # 🔥 Instead of error, return to assistant

graph.add_node("tools", tools_node)

# ✅ ISS Locator Agent Node
@traceable
def iss_agent_node(state: State) -> dict:
    location_info = get_iss_location(state["user_input"])  # Fetch ISS Location
    return {"agent_response": location_info}  # ✅ Return structured response

graph.add_node("iss_agent", iss_agent_node)

# ✅ Exit Node (Final Step)
@traceable
def exit_node(state: State) -> dict:
    """
    Formats the final response before exiting.
    """
    user_question = state["user_input"]
    agent_response = state.get("agent_response", "No response available.")

    prompt = f"""
    You are an AI assistant. A user asked about space, and an AI agent retrieved an answer.
    Please generate a clear, concise, and informative response.

    User Question: "{user_question}"
    AI Agent's Response: "{agent_response}"

    Generate a user-friendly response.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        final_answer = response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"⚠️ OpenAI API Error in exit_node: {e}")
        final_answer = "⚠️ Unable to generate a response at this time."

    return {"final_response": final_answer}  # ✅ Return structured final response

graph.add_node("exit_node", exit_node)

# ✅ Set Entry Point
graph.set_entry_point("assistant")

# ✅ Routing Paths
graph.add_conditional_edges(
    "assistant",
    lambda state: state["next_node"],
    {
        "tools": "tools",
        "exit_node": "exit_node"
    }
)

graph.add_conditional_edges(
    "tools",
    lambda state: state["next_tool"],
    {
        "iss_agent": "iss_agent",
        "assistant": "assistant"  # 🔥 Instead of exiting, go back to Assistant
    }
)

# ✅ Ensure Tool Execution Always Loops Back to Tools (or Allows Exit)
graph.add_conditional_edges(
    "iss_agent",
    lambda state: "tools",  # Loop back to tools for more queries
    {
        "tools": "tools"
    }
)

# ✅ Set Exit Node as the Final Step
graph.set_finish_point("exit_node")

# ✅ Compile the Graph
compiled_graph = graph.compile()
logging.info("🚀 LangGraph Compiled Successfully with Assistant-First Exit")