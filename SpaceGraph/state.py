from typing import TypedDict, Optional

class State(TypedDict):
    user_input: str  # Stores the user's question
    agent_response: Optional[str]  # Stores the AI agent's response
