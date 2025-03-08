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
    user_input = state["user_input"].lower()  # Normalize input for easier matching

    # Define a routing prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are an AI router responsible for determining which AI agent should handle a user's request. "
        "Given the user's input, return ONLY one of the following without quotes:\n"
        "- iss_locator_node (if the user asks about the ISS location)\n"
        "- astros_in_space_node (if the user asks about astronauts in space)\n"
        "- weather_node (if the user asks about weather, temperature, or conditions)\n\n"
        "If the user asks for weather *specifically at the ISS location*, return 'iss_locator_node' first, "
        "so it can fetch ISS coordinates before fetching weather."
        ),
        ("human", "{user_input}")
    ])
    
    # Run LLM with user input
    response = llm.invoke(prompt.format(user_input=user_input))

    # Extract response as a string
    if hasattr(response, "content"):
        response_text = response.content.strip()
    elif isinstance(response, str):
        response_text = response.strip()
    else:
        response_text = "iss_locator_node"  # Default fallback

    # ✅ Fix: Ensure ISS Weather Requests are Handled Correctly
    if "weather" in user_input and "iss" in user_input:
        logging.info("🌦️ User requested ISS weather. Routing to ISS Locator first.")
        state["next_agent"] = "iss_locator_node"
        state["weather_requested"] = True  # ✅ Flag for ISS Locator to continue to weather_node

    else:
        state["next_agent"] = response_text
        state["weather_requested"] = False  # ✅ Ensure normal routing

    state["next_step"] = state["next_agent"]  # ✅ Ensure routing function picks it up
    logging.info(f"📡 Routing user input '{user_input}' to: {state['next_agent']}")

    return state
