import os
import operator
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver # saves in RAM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

load_dotenv()

from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] # langgraph builtin function

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# gemini_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0.2,
# )

def chat_node(state: State):
    # take user query from state
    messages = state["messages"]
    
    # send to llm
    res = llm.invoke(messages)

    # return response
    return {
        'messages': [res]
    }


def name_chat(state: State):
    messages = state["messages"]
    # extract first user message
    for msg in messages:
        if isinstance(msg, HumanMessage):
            first_user_msg = msg.content
            break

checkpointer = MemorySaver()
graph = StateGraph(State)

# nodes
graph.add_node("chat_node", chat_node)

# edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# compile
chatbot = graph.compile(checkpointer=checkpointer)

def generate_thread_topic(first_message: str) -> str:
    """Generate a short topic/title for the thread based on the first human message."""
    prompt = f"""Generate a very short topic title (3-6 words max) for a conversation that starts with this message. 
    Return ONLY the title, no quotes, no explanation.

    Message: {first_message}

    Title:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()
