from typing import TypedDict, Optional, Dict, List, Any

class State(TypedDict):
    user_input: str  # Stores the original user question
    tool_responses: Dict[str, str]  # Stores each tool's response
    final_response: Optional[str]  # Stores the assistant's final formatted response
    pending_tools: List[str]  # Keeps track of which tools still need to run
    parameters: Dict[str, Any]  # ✅ Store extracted tool parameters (e.g., {"weather_agent": {"city": "Toronto"}})
    apod_url: Optional[str]  # Stores the APOD image URL
    apod_text: Optional[str]  # Stores the APOD explanation