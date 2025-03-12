import time
import requests
import streamlit as st
import json

# Set API URL
API_URL = "http://spacegraph:2024"

# Streamlit UI
st.set_page_config(page_title="SpaceGraph AI", layout="centered")
st.title("🌌 SpaceGraph AI Assistant")

# User input
user_input = st.text_input("Ask a question about space:", "")

# Function to create a new thread
def create_thread():
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(f"{API_URL}/threads", headers=headers, json={})
    if response.status_code == 200:
        return response.json().get("thread_id")
    return None

# Function to send query
def send_query(thread_id, query):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "assistant_id": "space_agent",
        "input": {"user_input": query}
    }
    response = requests.post(f"{API_URL}/threads/{thread_id}/runs/stream", headers=headers, json=data, stream=True)
    return response

# Function to clean and extract relevant data
def extract_final_response(response):
    final_response = None
    agents_used = {}

    for chunk in response.iter_lines():
        if chunk:
            decoded_line = chunk.decode('utf-8')
            try:
                data = json.loads(decoded_line.replace("data: ", "", 1))  # Clean "data: " prefix if present

                # Capture AI Agent response
                if "tool_responses" in data:
                    for agent, agent_data in data["tool_responses"].items():
                        if "agent_response" in agent_data:
                            agents_used[agent] = agent_data["agent_response"]["agent_response"]

                # Capture final response
                if "final_response" in data:
                    final_response = data["final_response"]

            except json.JSONDecodeError:
                pass  # Ignore non-JSON data

    return agents_used, final_response

if user_input:
    with st.spinner("🚀 Thinking..."):
        thread_id = create_thread()
        if thread_id:
            response = send_query(thread_id, user_input)
            if response.status_code == 200:
                agents_used, final_response = extract_final_response(response)

                # Show the agents that were invoked
                if agents_used:
                    st.subheader("🤖 AI Agents Invoked:")
                    for agent, answer in agents_used.items():
                        st.write(f"**{agent.replace('_', ' ').title()}**: {answer}")

                # Highlight the final answer
                if final_response:
                    st.subheader("✅ Final Answer:")
                    st.success(final_response)
                else:
                    st.warning("⚠️ No final answer was generated.")

            else:
                st.error("❌ Failed to get a response from SpaceGraph API")
        else:
            st.error("❌ Could not create a conversation thread.")
