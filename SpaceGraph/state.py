from typing import TypedDict, Optional, List

class ISSLocation(TypedDict):
    latitude: str
    longitude: str

class Astronaut(TypedDict):
    name: str
    craft: str

class WeatherData(TypedDict):  # ✅ NEW: Add Weather Data Structure
    temperature_c: Optional[float]
    condition: Optional[str]
    humidity: Optional[int]
    wind_kph: Optional[float]
    error: Optional[str]

class State(TypedDict):
    user_input: str
    iss_location: Optional[ISSLocation]
    astronauts: Optional[List[Astronaut]]
    weather: Optional[WeatherData]  # ✅ Store Weather Data
    weather_requested: bool  # ✅ Flag to track if weather at ISS was requested
    next_agent: Optional[str]
    final_answer: Optional[str]  # ✅ Ensure final answer is stored for UI access