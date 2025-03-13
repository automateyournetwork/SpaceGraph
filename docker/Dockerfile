FROM python:3.12-slim

WORKDIR /SpaceGraph

RUN pip install langchain_openai

RUN pip install fastapi

RUN pip install langsmith

RUN pip install --upgrade "langgraph-cli[inmem]"

# Now copy the rest of the application files
COPY /SpaceGraph /SpaceGraph
COPY /scripts /scripts/

CMD ["/bin/bash", "/scripts/startup.sh"]