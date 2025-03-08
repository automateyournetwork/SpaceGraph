FROM python:3.12-slim

WORKDIR /SpaceGraph

# Copy only requirements.txt first
COPY ./SpaceGraph/requirements.txt /SpaceGraph/requirements.txt

# Install dependencies separately to leverage Docker cache
RUN pip install --no-cache-dir -r /SpaceGraph/requirements.txt

# Now copy the rest of the application files
COPY ./SpaceGraph /SpaceGraph

CMD ["uvicorn", "run_langgraph:app", "--host", "0.0.0.0", "--port", "8123"]
