import os
from typing import Literal
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import ToolException, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, MessagesState, StateGraph

# Load environment variables from .env file
load_dotenv(override=True)
# set API key for Google Gemini
api_key = os.getenv("GOOGLE_API_KEY")

def handle_error(error: Exception) -> str:
    # function runs when tool crashes
    return f"TOOL ERROR:  The following error occured: {error.args[0]}"

# our tools
@tool
def divide(a: int, b: int) -> float:
    """Divides a by b"""
    if 0 == b:
        raise ToolException("cannot divide by zero")
    return a/b

# manually bind the handler
divide.handle_tool_error = handle_error

@tool("multiply", description="Multiplies two numbers")
def multiply(a: int, b:int) -> int:
    """multiply a by b"""
    return a*b

tools = [multiply, divide]

# -- 2. setup agent components

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
llm_with_tools = llm.bind_tools(tools)

def reasoner(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response]
    }

def router(state: MessagesState) -> Literal["__end__", "tools"]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"
    return '__end__'

# -- 3. build the graph

builder = StateGraph(MessagesState)

builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", router)
builder.add_edge("tools", "reasoner")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

print("--- Starting Gemini Flash Agent ---")
config = {"configurable": {"thread_id": "test_gemini_1"}}

inputs = {
    "messages": [
        ("user", "What is 10 divided by 0?")
    ]
}

for event in graph.stream(inputs, config):
    for value in event.values():
        msg = value['messages'][-1]

        if hasattr(msg, "tool_call_id"):
            print(f"\n[Tool Output]: {msg.content}")
        elif hasattr(msg,'tool_calls') and len(msg.tool_calls) > 0:
            print(f"\n[Gemini Call]: {msg.tool_calls}")
        else:
            print(f"\n[GEMINI FINAL]: {msg.content}")
