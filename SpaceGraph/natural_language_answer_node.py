import json
import logging
from langchain_openai import ChatOpenAI
import os
from state import State

# Load API Key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM Model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=OPENAI_API_KEY
)

# Natural Language Answer Node
def natural_language_answer_node(state: State) -> State:
    """Converts structured response data into human-readable text using LLM."""
    
    # Log structured state data before processing
    logging.info(f"📡 Received Structured Data: {json.dumps(state, indent=2)}")

    # ✅ Extract key data from state
    iss_location = state.get("iss_location", {})
    astronauts = state.get("astronauts", [])
    weather_data = state.get("weather", {})

    structured_data = None  # Will store the formatted summary

    # ✅ Handle ISS Location
    if iss_location and "latitude" in iss_location and "longitude" in iss_location:
        logging.info("🛰️ Processing ISS Location Data")
        structured_data = (
            f"The International Space Station (ISS) is currently located at "
            f"Latitude: {iss_location['latitude']}, Longitude: {iss_location['longitude']}."
        )

    # ✅ Handle Astronaut Data
    if astronauts:
        logging.info("🚀 Processing Astronaut Data")
        astronaut_list = [f"{astro['name']} (aboard {astro['craft']})" for astro in astronauts]
        astronaut_text = ", ".join(astronaut_list)
        structured_data = f"There are currently {len(astronauts)} astronauts in space: {astronaut_text}."

    # ✅ Handle Weather Data (Either ISS Location, City, or Lat/Long)
    if weather_data and "current" in weather_data and "location" in weather_data:
        logging.info("🌦️ Processing Weather Data")

        location_name = weather_data["location"].get("name", "Unknown location")
        region = weather_data["location"].get("region", "")
        country = weather_data["location"].get("country", "")
        condition = weather_data["current"]["condition"]["text"]
        temperature_c = weather_data["current"]["temp_c"]
        feels_like_c = weather_data["current"]["feelslike_c"]
        humidity = weather_data["current"]["humidity"]
        wind_kph = weather_data["current"]["wind_kph"]

        # Check if this is ISS weather
        if iss_location and (weather_data["location"].get("lat") == float(iss_location["latitude"]) and
                             weather_data["location"].get("lon") == float(iss_location["longitude"])):
            logging.info("🌦️ Weather at ISS Detected")
            structured_data = (
                f"The current weather at the ISS location is {condition}. "
                f"The temperature is {temperature_c}°C, but it feels like {feels_like_c}°C. "
                f"Humidity is {humidity}% and wind speed is {wind_kph} km/h."
            )
        else:
            structured_data = (
                f"The weather in {location_name}, {region} {country} is currently {condition}. "
                f"The temperature is {temperature_c}°C, but it feels like {feels_like_c}°C. "
                f"Humidity is {humidity}% and wind speed is {wind_kph} km/h."
            )

    # ✅ Default Fallback Message
    if not structured_data:
        structured_data = "I'm sorry, but I couldn't retrieve the requested information."

    logging.info(f"📤 Sending Data to LLM: {structured_data}")

    # LLM Prompt
    prompt = f"""
    You are an AI assistant that translates structured information into **clear, human-friendly responses**.
    Given this structured information:

    "{structured_data}"

    Please write a **concise and engaging response** in full sentences.
    """
    
    # Invoke LLM
    response = llm.invoke(prompt)
    logging.info(f"📝 LLM Raw Response: {response}")

    # Store the response
    if hasattr(response, "content") and response.content:
        state["final_answer"] = response.content.strip()
    else:
        state["final_answer"] = "I'm sorry, but I couldn't generate a response."

    logging.info(f"✅ Final Processed Answer: {state['final_answer']}")
    
    return state
