#!/bin/bash
echo "🚀 Starting LangGraph FastAPI Server..."
cd /SpaceGraph
uvicorn run_langgraph:app --host 0.0.0.0 --port 8123
