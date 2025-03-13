import os
import logging
from langchain_openai import ChatOpenAI  # ✅ FIXED: Use LangChain's OpenAI wrapper
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

# Tools (Assume they are implemented elsewhere)
from neo import get_near_earth_objects
from apod import get_apod
from weather import get_weather
from iss_locator import get_iss_location
from astronauts_in_space import get_astronauts

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ✅ Load API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Define Available Tools
tools = [
    get_iss_location,
    get_astronauts,
    get_weather,
    get_apod,
    get_near_earth_objects
]

# ✅ Use ChatOpenAI (Now Supports `.bind_tools`)
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# ✅ Bind Tools to LLM
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# ✅ System Message
sys_msg = SystemMessage(content="You are a helpful assistant providing space-related information.")

# ✅ Define Assistant Node (Now Matches LangGraph Docs)
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# ✅ Build the LangGraph
builder = StateGraph(MessagesState)

# ✅ Add Nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# ✅ Define Edges (Matches Docs)
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,  # Routes to "tools" if tools are needed, else to END
)
builder.add_edge("tools", "assistant")  # ✅ Tools always return to assistant

# ✅ Compile the Graph
compiled_graph = builder.compile()

logger.info("🚀 Space Information LangGraph compiled successfully")
