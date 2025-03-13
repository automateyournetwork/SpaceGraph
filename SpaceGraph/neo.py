import os
import time
import logging
import requests
from dotenv import load_dotenv
from langsmith import traceable

# Load environment variables
load_dotenv()
NEO_API_URL = "https://api.nasa.gov/neo/rest/v1/feed"
NASA_API_KEY = os.getenv("NASA_API_KEY")  # ✅ Ensure you set this in your .env file

# Configure logging
logging.basicConfig(level=logging.INFO)

class NEOTracker:
    def __init__(self):
        self.api_url = NEO_API_URL
        self.api_key = NASA_API_KEY

    @traceable
    def get_near_earth_objects(self):
        """
        Fetches data on Near Earth Objects (NEOs) for the current day.

        Returns:
            dict: A structured response containing information about the closest asteroids.
        """
        params = {"api_key": self.api_key}
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.get(self.api_url, params=params, timeout=5)
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
                    response_text = "No near-earth objects detected today."

                return {"agent_response": response_text}

            except requests.exceptions.RequestException as e:
                logging.error(f"❌ NEO API Request Failed (Attempt {attempt+1}): {e}")
                time.sleep(2)

        return {"agent_response": "⚠️ Unable to retrieve NEO data after multiple attempts."}

@traceable
def get_near_earth_objects(state):
    """
    LangGraph-compatible function to fetch Near Earth Object (NEO) data.

    Args:
        state (dict): The current state of the conversation.

    Returns:
        dict: The agent response containing the NEO data.
    """
    tracker = NEOTracker()
    return tracker.get_near_earth_objects()
