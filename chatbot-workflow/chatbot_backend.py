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

checkpointer = MemorySaver()
graph = StateGraph(State)

# nodes
graph.add_node("chat_node", chat_node)

# edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# compile
chatbot = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    flag: bool = True
    thread_id = 1
    while flag:
        usr_msg = input("USER\t>")

        if usr_msg.strip().lower() in ['exit', 'quit', 'bye']:
            flag = False
            print("Exited.")
            continue

        print('USER\t:', usr_msg)

        config = {
            'configurable': {
                'thread_id': thread_id
            }
        }

        res = chatbot.invoke({
            'messages': [HumanMessage(content=usr_msg)]
        }, config=config)

        print('STELLA\t:', res['messages'][-1].content)

    chatbot.get_state(config=config)
