import os
import logging
import requests
from dotenv import load_dotenv
from langsmith import traceable

# ✅ Load Environment Variables
load_dotenv()
NEO_API_URL = "https://api.nasa.gov/neo/rest/v1/feed"
NASA_API_KEY = os.getenv("NASA_API_KEY")  # ✅ Ensure this is set in your .env file

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO)

@traceable  # ✅ Enable LangSmith tracing
def get_near_earth_objects(messages):
    """
    Fetches data on Near Earth Objects (NEOs) for the current day.

    Parameters:
    - messages (list): The message history (not directly used, but required by LangGraph).

    Returns:
    - dict: A structured response containing information about the closest asteroids.

    Example Output:
    ```
    {
        "agent_response": "🛰️ Asteroid 123 - Size: 100m - Miss Distance: 500,000km - Velocity: 50,000kph - Hazardous: No"
    }
    ```
    """
    params = {"api_key": NASA_API_KEY}

    try:
        response = requests.get(NEO_API_URL, params=params, timeout=5)  # ✅ Timeout for safety
        response.raise_for_status()
        data = response.json()

        # ✅ Extract relevant asteroid data
        near_earth_objects = data.get("near_earth_objects", {})
        closest_objects = []

        for date, asteroids in near_earth_objects.items():
            for asteroid in asteroids:
                closest_objects.append({
                    "name": asteroid["name"],
                    "size": asteroid["estimated_diameter"]["meters"]["estimated_diameter_max"],
                    "miss_distance_km": asteroid["close_approach_data"][0]["miss_distance"]["kilometers"],
                    "velocity_kph": asteroid["close_approach_data"][0]["relative_velocity"]["kilometers_per_hour"],
                    "hazardous": asteroid["is_potentially_hazardous_asteroid"]
                })

        # ✅ Sort by closest distance
        closest_objects = sorted(closest_objects, key=lambda x: float(x["miss_distance_km"]))

        if closest_objects:
            response_text = "\n".join([
                f"🛰️ {obj['name']} - Size: {obj['size']:.2f}m - Miss Distance: {obj['miss_distance_km']}km - Velocity: {obj['velocity_kph']}kph - Hazardous: {'Yes' if obj['hazardous'] else 'No'}"
                for obj in closest_objects[:5]  # Limit to 5 results
            ])
        else:
            response_text = "🛰️ No near-earth objects detected today."

        return {"agent_response": response_text}

    except requests.exceptions.Timeout:
        logging.error("❌ NEO API Request Timed Out.")
        return {"agent_response": "⚠️ The request timed out. Please try again later."}

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ NEO API Request Failed: {e}")
        return {"agent_response": "⚠️ Unable to retrieve NEO data at this time."}
