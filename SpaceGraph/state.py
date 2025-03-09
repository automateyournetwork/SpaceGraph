from typing import TypedDict, Optional, List
import datetime

class ISSLocation(TypedDict):
    latitude: str
    longitude: str

class Astronaut(TypedDict):
    name: str
    craft: str

class WeatherData(TypedDict):
    temperature_c: Optional[float]
    condition: Optional[str]
    humidity: Optional[int]
    wind_kph: Optional[float]
    error: Optional[str]

class APODData(TypedDict):
    title: Optional[str]
    description: Optional[str]
    date: Optional[str]
    media_type: Optional[str]
    url: Optional[str]
    error: Optional[str]

class Asteroid(TypedDict):  # ✅ NEW: Structure for individual asteroids
    name: str
    diameter_km: float
    velocity_kph: str
    miss_distance_km: str
    hazardous: bool

class NEOData(TypedDict):  # ✅ NEW: Store Near-Earth Object Data
    date: str
    asteroids: List[Asteroid]
    error: Optional[str]

class State(TypedDict):
    user_input: str
    location: Optional[str]  # ✅ Store city name or lat/long
    iss_location: Optional[ISSLocation]
    astronauts: Optional[List[Astronaut]]
    weather: Optional[WeatherData]  # ✅ Store Weather Data
    apod: Optional[APODData]  # ✅ Store APOD Data
    neo: Optional[NEOData]  # ✅ NEW: Store Near-Earth Object Data
    weather_requested: bool  # ✅ Track if weather at ISS was requested
    next_agent: Optional[str]
    final_answer: Optional[str]  # ✅ Ensure final answer is stored for UI access
    timestamp: str  # ✅ Store when the state was last updated
