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

    # Extract relevant data from state
    astronauts = state.get("astronauts", [])

    # Format astronaut data into a readable string before sending to LLM
    if astronauts:
        astronaut_list = [f"{astro['name']} (aboard {astro['craft']})" for astro in astronauts]
        astronaut_text = ", ".join(astronaut_list)
        structured_data = f"There are currently {len(astronauts)} astronauts in space: {astronaut_text}."
    else:
        structured_data = "I couldn't retrieve astronaut information at this time."

    # Log data being sent to the LLM
    logging.info(f"📤 Sending Data to LLM: {structured_data}")

    # LLM Prompt
    prompt = f"""
    You are an AI assistant that translates structured information into **clear, human-friendly responses**.
    Given this structured information:
    
    "{structured_data}"
    
    Please write a **concise and engaging response** in full sentences.
    """
    
    # Invoke LLM and log raw response
    response = llm.invoke(prompt)
    logging.info(f"📝 LLM Raw Response: {response}")

    # Ensure response is processed correctly
    if hasattr(response, "content") and response.content:
        state["final_answer"] = response.content.strip()
    else:
        state["final_answer"] = "I'm sorry, but I couldn't generate a response."

    # ✅ Log final answer
    logging.info(f"✅ Final Processed Answer: {state['final_answer']}")
    
    return state
