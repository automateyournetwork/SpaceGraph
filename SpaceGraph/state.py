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

class Asteroid(TypedDict):
    name: str
    diameter_km: float
    velocity_kph: str
    miss_distance_km: str
    hazardous: bool

class NEOData(TypedDict):
    date: str
    asteroids: List[Asteroid]
    error: Optional[str]

class MarsWeatherData(TypedDict):  # ✅ NEW: Store Mars Weather Data
    sol: str  # Martian day
    season: str
    temperature_c: Optional[float]
    pressure_pa: Optional[float]
    wind_speed_mps: Optional[float]
    wind_direction: Optional[str]
    first_utc: Optional[str]
    last_utc: Optional[str]
    error: Optional[str]

class State(TypedDict):
    user_input: str
    location: Optional[str]  # ✅ Store city name or lat/long
    iss_location: Optional[ISSLocation]
    astronauts: Optional[List[Astronaut]]
    weather: Optional[WeatherData]  # ✅ Store Earth Weather Data
    apod: Optional[APODData]  # ✅ Store APOD Data
    neo: Optional[NEOData]  # ✅ Store Near-Earth Object Data
    mars_weather: Optional[MarsWeatherData]  # ✅ NEW: Store Mars Weather Data
    weather_requested: bool  # ✅ Track if weather at ISS was requested
    next_agent: Optional[str]
    final_answer: Optional[str]  # ✅ Ensure final answer is stored for UI access
    timestamp: str  # ✅ Store when the state was last updated
