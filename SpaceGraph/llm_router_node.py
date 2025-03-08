from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from state import State
import logging

# Load API Key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM Model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=OPENAI_API_KEY
)

# Define LLM-powered router
def llm_router_node(state: State) -> State:
    user_input = state["user_input"].strip()  # Normalize input

    # Define a routing prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are an AI router responsible for determining which AI agent should handle a user's request. "
        "Given the user's input, return ONLY one of the following without quotes:\n"
        "- iss_locator_node (if the user asks about the ISS location)\n"
        "- astros_in_space_node (if the user asks about astronauts in space)\n"
        "- weather_node (if the user asks about weather, temperature, or conditions)\n\n"
        "If the user asks for weather, analyze whether they provided:\n"
        "1. A city name (like 'Toronto')\n"
        "2. A latitude/longitude pair (like '40.7,-74.0')\n\n"
        "If the user provided a city, return 'weather_node|CITY_NAME'.\n"
        "If they provided lat/long, return 'weather_node|LAT,LONG'.\n"
        "If they asked for weather at the ISS location, return 'iss_locator_node'."
        ),
        ("human", "{user_input}")
    ])
    
    # Run LLM with user input
    response = llm.invoke(prompt.format(user_input=user_input))

    # Extract response
    if hasattr(response, "content"):
        response_text = response.content.strip()
    elif isinstance(response, str):
        response_text = response.strip()
    else:
        response_text = "iss_locator_node"  # Default fallback

    # ✅ Handle ISS weather request (ensures location is fetched first)
    if "weather" in user_input.lower() and "iss" in user_input.lower():
        logging.info("🌦️ User requested ISS weather. Routing to ISS Locator first.")
        state["next_agent"] = "iss_locator_node"
        state["weather_requested"] = True  
        state["location"] = None  # No manual location, ISS will provide it

    # ✅ Handle general weather queries
    elif "weather_node" in response_text:
        logging.info("🌦️ User requested weather data.")
        state["next_agent"] = "weather_node"

        # Extract detected location (City or Lat/Long)
        parts = response_text.split("|")
        if len(parts) > 1:
            detected_location = parts[1].strip()
            state["location"] = detected_location  # Store city or coordinates

        state["weather_requested"] = False  

    else:
        state["next_agent"] = response_text
        state["weather_requested"] = False  

    state["next_step"] = state["next_agent"]  
    logging.info(f"📡 Routing user input '{user_input}' to: {state['next_agent']} with location {state.get('location', 'None')}")

    return state
