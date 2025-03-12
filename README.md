# SpaceGraph
LangGraph AI Agent approach with Space APIs

## Git Clone the Repo:

## Install Docker Desktop (Windows; WSL Linux; Mac)

## Docker-Compose Up

## Visit Various URLs: 

### Documentation

### LangGraph Studio

### REST API

#### REST API Instructions:
📖 Space Agent API Usage Guide

1️⃣ Create a New Thread
Each interaction starts by creating a thread.

``` bash
curl -X POST "http://127.0.0.1:2024/threads" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_generated_token" \
     -d '{}'
```

✅ Response Example:

``` json
{
    "thread_id": "99f5f9c9-c297-4510-9b3e-65535987149f"
}
```

3️⃣ Run a Query (e.g., Where is the ISS?)

Once a thread is created, use the thread ID to send queries.

``` bash
curl -X POST "http://127.0.0.1:2024/threads/YOUR_THREAD_ID/runs/stream" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_generated_token" \
     -d '{"assistant_id": "space_agent", "input": {"user_input": "Where is the ISS right now?"}}'
```

✅ Response Example:

``` json
{
    "user_input": "Where is the ISS right now?",
    "tool_responses": {
        "iss_agent": {
            "agent_response": "The ISS is currently at Latitude: -5.0048, Longitude: 59.0041."
        }
    },
    "final_response": "As of the latest update, the ISS is at latitude -5.0048 and longitude 59.0041. Its position changes every 90 minutes!"
}
```

4️⃣ Example Queries

You can modify the "user_input" field for different questions.

📌 Get the List of Astronauts in Space

``` bash
curl -X POST "http://127.0.0.1:2024/threads/YOUR_THREAD_ID/runs/stream" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_generated_token" \
     -d '{"assistant_id": "space_agent", "input": {"user_input": "Who is in space right now?"}}'
```

📌 Get the Current Weather in a City

``` bash
curl -X POST "http://127.0.0.1:2024/threads/YOUR_THREAD_ID/runs/stream" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_generated_token" \
     -d '{"assistant_id": "space_agent", "input": {"user_input": "What is the weather in Toronto right now?"}}'
```

📌 Get NASA’s Astronomy Picture of the Day (APOD)

``` bash
curl -X POST "http://127.0.0.1:2024/threads/YOUR_THREAD_ID/runs/stream" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_generated_token" \
     -d '{"assistant_id": "space_agent", "input": {"user_input": "Show me NASA's Astronomy Picture of the Day."}}'
```

🔁 Full API Workflow

Get an API token

Create a thread

Use the thread ID to send queries

This ensures proper session handling and maintains conversation context.