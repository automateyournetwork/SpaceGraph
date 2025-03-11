from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from state import State
import logging
from langsmith import traceable
from langsmith.wrappers import wrap_openai

# Load API Key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Wrap OpenAI client for LangSmith tracing
client = wrap_openai(OpenAI(api_key=OPENAI_API_KEY))

# ✅ Enable LangSmith Tracing if configured
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

if LANGSMITH_TRACING:
    logging.info("✅ LangSmith Tracing Enabled for `llm_router_node`")

@traceable  # 🚀 Auto-trace this function
def llm_router_node(state: State) -> State:
    """LLM-based router to determine which agent should handle a user's request."""

    user_input = state["user_input"].strip()  # Normalize input
    logging.info(f"📡 Received user input: '{user_input}'")

    # ✅ Ensure Mars weather is correctly routed before LLM processing
    if "weather" in user_input.lower() and "mars" in user_input.lower():
        logging.info("🌌 Overriding to Mars Weather Node.")
        state["next_agent"] = "mars_weather_node"
        return state  # ✅ Prevents unnecessary LLM call

    # ✅ Define a routing prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are an AI router responsible for determining which AI agent should handle a user's request. "
        "Given the user's input, return ONLY one of the following without quotes:\n"
        "- iss_locator_node (if the user asks about the ISS location)\n"
        "- astros_in_space_node (if the user asks about astronauts in space)\n"
        "- weather_node (if the user asks about weather, temperature, or conditions on Earth)\n"
        "- mars_weather_node (if the user asks about Mars weather, the weather on Mars, or conditions on Mars. If 'Mars' is mentioned in a weather request, return 'mars_weather_node')\n"
        "- apod_node (if the user asks about NASA's Astronomy Picture of the Day)\n"
        "- neo_node (if the user asks about asteroids, near-Earth objects, or potential asteroid impacts)\n\n"
        "If the user asks for weather on Earth, analyze whether they provided:\n"
        "1. A city name (like 'Toronto')\n"
        "2. A latitude/longitude pair (like '40.7,-74.0')\n\n"
        "If the user provided a city, return 'weather_node|CITY_NAME'.\n"
        "If they provided lat/long, return 'weather_node|LAT,LONG'.\n"
        "If they asked for weather at the ISS location, return 'iss_locator_node'."
        ),
        ("human", "{user_input}")
    ]).format(user_input=user_input)
    
    # ✅ Run LLM with user input using OpenAI client
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    # ✅ Extract response text safely
    response_text = response.choices[0].message.content.strip() if response.choices else "natural_language_answer_node"

    logging.info(f"🤖 Raw LLM Response: {response_text}")

    # ✅ Validate LLM response
    valid_nodes = {
        "iss_locator_node", "astros_in_space_node", "weather_node",
        "mars_weather_node", "apod_node", "neo_node"
    }
    if response_text not in valid_nodes:
        logging.warning(f"⚠️ Invalid routing response: {response_text}. Defaulting to natural_language_answer_node.")
        response_text = "natural_language_answer_node"

    # ✅ Handle ISS weather request
    if "weather" in user_input.lower() and "iss" in user_input.lower():
        logging.info("🌦️ User requested ISS weather. Routing to ISS Locator first.")
        state["next_agent"] = "iss_locator_node"
        state["weather_requested"] = True  
        state["location"] = None  # No manual location, ISS will provide it
    # ✅ Handle general weather queries
    elif "weather_node" in response_text:
        logging.info("🌦️ User requested weather data.")
        state["next_agent"] = "weather_node"
        
        # ✅ Extract detected location (City or Lat/Long)
        parts = response_text.split("|")
        if len(parts) > 1:
            detected_location = parts[1].strip()
            state["location"] = detected_location  # Store city or coordinates

        state["weather_requested"] = False  
    # ✅ Handle APOD request
    elif response_text == "apod_node":
        logging.info("🛰️ User requested NASA's Astronomy Picture of the Day.")
        state["next_agent"] = "apod_node"
    # ✅ Handle NEO request (Near-Earth Objects)
    elif response_text == "neo_node":
        logging.info("☄️ User requested Near-Earth Object (NEO) data.")
        state["next_agent"] = "neo_node"
    # ✅ Handle Mars weather request
    elif response_text == "mars_weather_node":
        logging.info("🪐 User requested Mars weather.")
        state["next_agent"] = "mars_weather_node"
    else:
        state["next_agent"] = response_text
        state["weather_requested"] = False  

    state["next_step"] = state["next_agent"]  

    logging.info(f"✅ Final Routing Decision: {state['next_agent']} (LLM suggested: {response_text})")

    return state