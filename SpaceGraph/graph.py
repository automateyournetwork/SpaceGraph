#Base Imports
import os
import re
import json
import openai
import logging

#Tools and State
from state import State
from neo import get_near_earth_objects
from apod import get_apod
from weather import get_weather
from iss_locator import get_iss_location
from astronauts_in_space import get_astronauts

#LANGSMITH
from langsmith import traceable
from langsmith.wrappers import wrap_openai

#LangGraph 
from langgraph.graph import StateGraph, END

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
    "weather_agent": get_weather,
    "apod_agent": get_apod,
    "neo_agent": get_near_earth_objects
}

@traceable
def assistant(state: State) -> dict:
    """
    Central node that processes user input, determines tools, and extracts structured parameters.
    """
    user_input = state["user_input"]
    tool_responses = state.get("tool_responses", {})
    pending_tools = state.get("pending_tools", [])

    # Ensure `assistant` always returns a dict
    if not isinstance(state, dict):
        return {"error": "Invalid state format"}

    # If tools are pending, move to `tools`
    if pending_tools:
        return {"next_node": "tools"}

    # If we have tool responses but no pending tools, move to `end`
    if tool_responses and not pending_tools:
        return {"next_node": "end"}

    # Process the input and determine necessary tools
    prompt = f"""
    You are an AI assistant that classifies user questions and extracts structured data.

    Available tools:
    - iss_agent: For ISS location, orbit, or tracking
    - astronauts_agent: For who is in space
    - weather_agent: For weather in a city or at lat/lon
    - apod_agent: For NASA Astronomy Picture of the Day
    - neo_agent: Near Earth Objects agent for asteroids near Earth

    Respond with a JSON object **inside triple backticks** like this:
    ```
    {{
        "tools": ["tool_name1", "tool_name2"],
        "parameters": {{}}
    }}
    ```
    If no tools apply, return:
    ```
    {{"tools": []}}
    ```

    User Question: "{user_input}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"🔍 Raw LLM Response: {response_text}")

        # Extract JSON from response
        json_match = re.search(r"```(?:json)?\n(.*?)\n```", response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)

        structured_response = json.loads(response_text)  # ✅ Use json.loads()
        logger.info(f"🔍 Parsed LLM JSON: {structured_response}")

        tools = structured_response.get("tools", [])
        parameters = structured_response.get("parameters", {})

        valid_tools = [tool for tool in tools if tool in AVAILABLE_TOOLS]

        if not valid_tools:
            return {"next_node": "end"}

        return {
            "pending_tools": valid_tools,
            "tool_responses": {},
            "parameters": parameters,
            "next_node": "tools"
        }

    except json.JSONDecodeError as e:
        logger.error(f"⚠️ JSON parsing error: {e}")
        return {"next_node": "end"}

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
        elif current_tool == "apod_agent":
            tool_result = get_apod(user_input)
        elif current_tool == "neo_agent":
            tool_result = get_near_earth_objects(user_input)
        elif current_tool == "weather_agent":
            # ✅ Extract city from state parameters
            weather_params = state.get("parameters", {})
            city = weather_params.get("location", None)  # Fix parameter lookup

            if city:
                tool_result = get_weather({"city": city})
            else:
                logger.error("⚠️ Weather tool failed: Missing city parameter.")
                tool_result = "⚠️ Missing city or coordinates for weather lookup."
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
    weather_data = tool_responses.get("weather_agent", "No weather data was requested or available.")
    apod_data = tool_responses.get("apod_agent", "No APOD data was requested or available")
    neo_data = tool_responses.get("neo_agent", "No NEO data was requested or available")
    
    # Prepare comprehensive context for the LLM
    context = ""
    
    # Only include tool data if it was actually retrieved
    if "iss_agent" in tool_responses:
        context += f"ISS LOCATION DATA:\n{iss_location}\n\n"
    
    if "astronauts_agent" in tool_responses:
        context += f"ASTRONAUTS IN SPACE DATA:\n{astronauts_data}\n\n"

    if "weather_agent" in tool_responses:
        context += f"WEATHER ON EARTH DATA:\n{weather_data}\n\n"

    if "apod_agent" in tool_responses:
        context += f"APOD DATA:\n{apod_data}\n\n"

    if "neo_agent" in tool_responses:
        context += f"Near Earth Object DATA:\n{neo_data}\n\n"

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
    
    Your response should be conversational, accurate, and engaging. You have data from one or more agents. Please try to make correlations and connections
    between the various AI Agent responses. 
    
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
    
    return {"final_response": final_answer, "next_node": END}  # ✅ Always return a dict

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