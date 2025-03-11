import os
import logging
import openai
from langgraph.graph import StateGraph, END
from state import State
from iss_locator import get_iss_location
from astronauts_in_space import get_astronauts
from langsmith.wrappers import wrap_openai
from langsmith import traceable

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Wrap OpenAI client with LangSmith tracing
client = wrap_openai(openai.Client())

# Initialize LangGraph StateGraph
graph = StateGraph(State)

# Define Available Tools
AVAILABLE_TOOLS = {
    "iss_agent": get_iss_location,
    "astronauts_agent": get_astronauts,
}

@traceable
def assistant(state: State) -> dict:
    """
    Central node that processes user input, determines tools, and handles routing.
    """
    user_input = state["user_input"]
    tool_responses = state.get("tool_responses", {})
    pending_tools = state.get("pending_tools", [])
    
    # If we have pending tools, continue routing to tools
    if pending_tools:
        logger.info(f"🔄 Routing to tools: {pending_tools}")
        return {"next_node": "tools"}
    
    # If we have tool responses but no pending tools, we're done with tools
    if tool_responses and not pending_tools:
        logger.info("✅ All tools completed. Generating final response.")
        return {"next_node": "end"}
    
    # First time through - classify user question
    prompt = f"""
    You are an AI assistant that classifies user questions about space.
    Identify which tools are needed to answer this space-related question.

    Available tools:
    - iss_agent: For questions about ISS location, orbit, tracking, or position
    - astronauts_agent: For questions about who is currently in space, astronaut details, or space crews
    
    If multiple tools are needed, return them as a comma-separated list.
    If no tools apply, simply return "none".

    User Question: "{user_input}"

    Your response should ONLY include tool names from the available list, or "none".
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        response_text = response.choices[0].message.content.strip()
        logger.info(f"🔍 Tool classifier response: {response_text}")
        
        # Parse the comma-separated tool names
        tools = [tool.strip() for tool in response_text.split(",")]
        # Filter out any invalid tool names
        valid_tools = [tool for tool in tools if tool in AVAILABLE_TOOLS]
        
        if not valid_tools or "none" in tools:
            logger.info("No applicable tools found, routing directly to end")
            return {"next_node": "end"}
        
        logger.info(f"🔍 Identified tools needed: {valid_tools}")
        return {"pending_tools": valid_tools, "tool_responses": {}, "next_node": "tools"}
        
    except Exception as e:
        logger.error(f"⚠️ Error in assistant routing: {e}")
        return {"next_node": "end"}

@traceable
def tools(state: State) -> dict:
    """
    Tools router that selects and calls the appropriate tool.
    """
    pending_tools = state.get("pending_tools", [])
    
    if not pending_tools:
        # Safety check - shouldn't happen with proper routing
        logger.warning("⚠️ Tools node called with no pending tools")
        return {"next_node": "assistant"}
    
    # Select the first pending tool
    current_tool = pending_tools[0]
    logger.info(f"🔧 Processing tool: {current_tool}")
    
    # Call the appropriate tool function
    user_input = state["user_input"]
    tool_responses = state.get("tool_responses", {})
    
    try:
        if current_tool == "iss_agent":
            tool_result = get_iss_location(user_input)
        elif current_tool == "astronauts_agent":
            tool_result = get_astronauts(user_input)
        else:
            logger.warning(f"⚠️ Unknown tool requested: {current_tool}")
            tool_result = "Error: Unknown tool requested."
            
        # Store the result
        tool_responses[current_tool] = tool_result
        logger.info(f"✅ Successfully processed {current_tool}")
        
    except Exception as e:
        logger.error(f"⚠️ Error processing {current_tool}: {e}")
        tool_responses[current_tool] = f"Error processing {current_tool}: {str(e)}"
    
    # Remove the tool from pending list regardless of success/failure
    pending_tools.remove(current_tool)
    
    # Return updated state
    return {
        "tool_responses": tool_responses,
        "pending_tools": pending_tools,
        "next_node": "assistant"  # Always return to assistant for routing
    }

@traceable
def end(state: State) -> dict:
    """
    Final node that generates a response based on tool results.
    """
    user_question = state["user_input"]
    tool_responses = state.get("tool_responses", {})
    
    # Extract tool responses with fallbacks for missing data
    iss_location = tool_responses.get("iss_agent", "No ISS location data was requested or available.")
    astronauts_data = tool_responses.get("astronauts_agent", "No astronaut data was requested or available.")
    
    # Prepare comprehensive context for the LLM
    context = ""
    
    # Only include tool data if it was actually retrieved
    if "iss_agent" in tool_responses:
        context += f"ISS LOCATION DATA:\n{iss_location}\n\n"
    
    if "astronauts_agent" in tool_responses:
        context += f"ASTRONAUTS IN SPACE DATA:\n{astronauts_data}\n\n"
    
    # If no tool was used, note that
    if not tool_responses:
        context = "No specific space data was retrieved for this query."
    
    prompt = f"""
    You are a helpful space information assistant. The user asked the following question:
    
    USER QUESTION: "{user_question}"
    
    Based on the retrieved data, provide a clear, informative, and natural-sounding response.
    Include key details from the data but organize them coherently.
    
    AVAILABLE DATA:
    {context}
    
    Your response should be conversational, accurate, and engaging. If both ISS location and 
    astronaut data are available, be sure to clearly connect the astronauts to the ISS when appropriate.
    
    If the data is incomplete, acknowledge limitations but be helpful with what's available.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        final_answer = response.choices[0].message.content.strip()
        logger.info("✅ Successfully generated final response")
    except Exception as e:
        logger.error(f"⚠️ Error generating final response: {e}")
        final_answer = "I'm sorry, I encountered an issue processing the space data. Please try again or rephrase your question."
    
    return {"final_response": final_answer, "next_node": END}

# Add nodes to the graph
graph.add_node("assistant", assistant)
graph.add_node("tools", tools)
graph.add_node("end", end)

# Set the entry point
graph.set_entry_point("assistant")

# Define edges
# Assistant can route to either tools or end
graph.add_conditional_edges(
    "assistant",
    lambda state: state.get("next_node"),
    {
        "tools": "tools",
        "end": "end"
    }
)

# Tools always return to assistant for re-evaluation
graph.add_edge("tools", "assistant")

# End node ends execution
graph.add_edge("end", END)

# Compile the graph
compiled_graph = graph.compile()
logger.info("🚀 Simplified Space Information LangGraph compiled successfully")