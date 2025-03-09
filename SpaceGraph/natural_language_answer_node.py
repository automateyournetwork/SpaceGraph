import json
import logging
import os
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from state import State

# Load API Key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Wrap OpenAI client for LangSmith tracing
client = wrap_openai(OpenAI(api_key=OPENAI_API_KEY))

# ✅ Enable LangSmith Tracing if configured
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

if LANGSMITH_TRACING:
    logging.info("✅ LangSmith Tracing Enabled for `natural_language_answer_node`")

@traceable  # 🚀 Auto-trace this function
def natural_language_answer_node(state: State) -> State:
    """Converts structured response data into human-readable text using LLM."""

    logging.info(f"📡 Received Structured Data: {json.dumps(state, indent=2)}")

    # ✅ Extract key data from state
    iss_location = state.get("iss_location", {})
    astronauts = state.get("astronauts", [])
    weather_data = state.get("weather", {})
    apod_data = state.get("apod", {})

    structured_data = None  # Will store the formatted summary

    # ✅ Handle APOD Data
    if apod_data and "title" in apod_data:
        logging.info("🛰️ Processing NASA APOD Data")
        
        title = apod_data.get("title", "Unknown Title")
        description = apod_data.get("description", "No description available.")
        date = apod_data.get("date", "Unknown Date")
        media_type = apod_data.get("media_type", "unknown")
        url = apod_data.get("url", "")

        if "error" in apod_data:
            structured_data = "I couldn't retrieve today's NASA Astronomy Picture of the Day."
        else:
            structured_data = (
                f"📷 NASA's Astronomy Picture of the Day for {date} is titled **{title}**.\n\n"
                f"**Description:** {description}\n\n"
                f"🔗 View it here: [NASA APOD]({url})"
            )

    # ✅ Handle ISS Location
    elif iss_location and "latitude" in iss_location and "longitude" in iss_location:
        logging.info("🛰️ Processing ISS Location Data")
        structured_data = (
            f"The International Space Station (ISS) is currently located at "
            f"Latitude: {iss_location['latitude']}, Longitude: {iss_location['longitude']}."
        )

    # ✅ Handle Astronaut Data
    elif astronauts:
        logging.info("🚀 Processing Astronaut Data")
        astronaut_list = [f"{astro['name']} (aboard {astro['craft']})" for astro in astronauts]
        astronaut_text = ", ".join(astronaut_list)
        structured_data = f"There are currently {len(astronauts)} astronauts in space: {astronaut_text}."

    # ✅ Handle Weather Data
    elif weather_data and "current" in weather_data and "location" in weather_data:
        logging.info("🌦️ Processing Weather Data")

        location_name = weather_data["location"].get("name", "Unknown location")
        region = weather_data["location"].get("region", "")
        country = weather_data["location"].get("country", "")
        condition = weather_data["current"]["condition"].get("text", "Unknown")
        temperature_c = weather_data["current"].get("temp_c", "N/A")
        feels_like_c = weather_data["current"].get("feelslike_c", "N/A")
        humidity = weather_data["current"].get("humidity", "N/A")
        wind_kph = weather_data["current"].get("wind_kph", "N/A")

        # Check if this is ISS weather
        if iss_location and (
            weather_data["location"].get("lat") == float(iss_location.get("latitude", 0)) and
            weather_data["location"].get("lon") == float(iss_location.get("longitude", 0))
        ):
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

    # ✅ LLM Prompt
    prompt = f"""
    You are an AI assistant that translates structured information into **clear, human-friendly responses**.
    Given this structured information:

    "{structured_data}"

    Please write a **concise and engaging response** in full sentences.
    """

    # ✅ Invoke LLM using OpenAI client
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    # ✅ Extract response safely
    response_text = response.choices[0].message.content.strip() if response.choices else "I'm sorry, but I couldn't generate a response."

    state["final_answer"] = response_text
    logging.info(f"✅ Final Processed Answer: {state['final_answer']}")

    return state
