import os
import json
from dotenv import load_dotenv
from typing import Any, Annotated
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from langgraph.runtime import Runtime
from langchain.agents import AgentState
from langchain.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain.agents.middleware import before_model
from langgraph.graph.message import REMOVE_ALL_MESSAGES

PostgresSaver = None
psycopg = None
try:
    from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore
    import psycopg  # type: ignore
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

load_dotenv()

# Database configuration
DB_URI = os.getenv("DATABASE_URL")
MODEL = os.getenv("MODEL", "None" )

# Use local Ollama model
llm = ChatOllama(
    model=MODEL,
    temperature=0.7
)

# Database helper function
def get_db_connection():
    """Get a database connection."""
    if DB_URI and psycopg:
        return psycopg.connect(DB_URI)
    return None

# Initialize users table if it doesn't exist
def init_users_table():
    """Create users table if it doesn't exist."""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id VARCHAR(50) PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        email VARCHAR(100),
                        balance DECIMAL(10, 2) DEFAULT 0
                    )
                """)
                # Insert sample data if table is empty
                cur.execute("SELECT COUNT(*) FROM users")
                if cur.fetchone()[0] == 0:
                    cur.execute("""
                        INSERT INTO users (user_id, name, email, balance) VALUES
                        ('user1', 'Alice', 'alice@example.com', 1000),
                        ('user2', 'Bob', 'bob@example.com', 2500)
                    """)
                conn.commit()
                print("✓ Users table initialized")
        except Exception as e:
            print(f"  Note: Users table setup: {e}")
        finally:
            conn.close()

# Initialize users table on startup
init_users_table()

# Define tools for the agent
@tool
def get_user_info(user_id: str) -> dict:
    """Retrieve user information by user ID from the database."""
    conn = get_db_connection()
    if not conn:
        return {"error": "Database connection not available"}
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, name, email, balance FROM users WHERE user_id = %s",
                (user_id,)
            )
            row = cur.fetchone()
            if row:
                return {
                    "user_id": row[0],
                    "name": row[1],
                    "email": row[2],
                    "balance": float(row[3])
                }
            return {"error": "User not found"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

@tool
def save_conversation(user_id: str, message: str, response: str) -> dict:
    """Save conversation to short-term memory."""
    return {"status": "saved", "user_id": user_id, "message_length": len(message)}

@tool
def retrieve_context(user_id: str) -> dict:
    """Retrieve recent conversation context for a user."""
    return {"user_id": user_id, "context": "Recent conversation history"}


# Define the state for the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last 5 messags in the converstion history"""

    messages = state["messages"]

    if(len(messages) < 5):
        return None
    
    # Keep only the last 5 messages
    trimmed_messages = messages[-5:]

    return {
        "messages": [RemoveMessage(
            id=REMOVE_ALL_MESSAGES), *trimmed_messages
        ]
    }

# Define tools list
tools = [get_user_info, save_conversation, retrieve_context]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

def reasoner(state: State):
    """Process messages and generate response using LLM with tools."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def router(state: AgentState) -> Annotated['tools', '__end__']:
    messages = state["messages"][-1];

    if hasattr(messages, 'tool_calls') and messages.tool_calls:
        return "tools"
    return '__end__';

# Build the graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("reasoner", reasoner)
graph_builder.add_node("tools", ToolNode(tools=tools))

# Add edges
graph_builder.add_edge(START, "reasoner")
graph_builder.add_conditional_edges("reasoner", router)
graph_builder.add_edge("tools", "reasoner")

# Initialize checkpointer for conversation memory
checkpointer = None
if POSTGRES_AVAILABLE and DB_URI and psycopg and PostgresSaver:
    try:
        # Use autocommit for setup to avoid transaction issues with Neon
        connection = psycopg.connect(DB_URI, autocommit=True)
        checkpointer = PostgresSaver(connection)
        try:
            checkpointer.setup()
        except Exception as setup_error:
            # Tables might already exist, that's okay
            print(f"  Note: {setup_error}")
        print("✓ PostgreSQL checkpointer initialized - conversations will persist to database")
    except Exception as e:
        print(f"✗ PostgreSQL connection failed: {e}")
        print("  Falling back to in-memory storage...")
        checkpointer = MemorySaver()
        print("✓ In-memory checkpointer initialized - conversations will be lost on restart")
else:
    checkpointer = MemorySaver()
    print("✓ In-memory checkpointer initialized - conversations will be lost on restart")
    if not POSTGRES_AVAILABLE:
        print("  (Install 'langgraph-checkpoint-postgres' and 'psycopg' for persistent storage)")

# Compile the graph
graph = graph_builder.compile(checkpointer=checkpointer)
print("✓ Agent with short-term memory created successfully")


if __name__ == "__main__":
    # Simulate a conversation
    config: RunnableConfig = {"configurable": {"thread_id": "user1_session"}}

    while(True):
        prompt = input(f"User\t> ")
        
        # Debug command to see conversation history
        if prompt.lower() == "/history":
            state = graph.get_state(config)
            messages = state.values.get("messages", [])
            print(f"\n--- Conversation History ({len(messages)} messages) ---")
            for i, msg in enumerate(messages):
                role = msg.type if hasattr(msg, 'type') else 'unknown'
                content = msg.content if hasattr(msg, 'content') else str(msg)
                # Truncate long messages
                if isinstance(content, str) and len(content) > 100:
                    content = content[:100] + "..."
                print(f"  [{i+1}] {role}: {content}")
            print("--- End of History ---\n")
            continue
        
        if prompt.lower() == "/quit":
            print("Goodbye!")
            break
        
        print(f"\nStella\t> thinking...", end="", flush=True)
        
        first_chunk = True
        for chunk in graph.stream({"messages": [("user", prompt)]}, config=config, stream_mode="messages"):
            msg, metadata = chunk
            node = metadata.get("langgraph_node") if hasattr(metadata, 'get') else None
            
            # Only process messages from reasoner node
            if node != "reasoner":
                continue
                
            # Check if this is a tool call (no content, has tool_calls)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                if first_chunk:
                    print("\r" + " " * 30 + "\r", end="", flush=True)
                    print(f"Stell\t> [using tools...]", end="", flush=True)
                    first_chunk = False
                continue
            
            # Process content
            if hasattr(msg, 'content') and msg.content:
                # Clear "thinking..." or "[using tools...]" on first response chunk
                if first_chunk:
                    print("\r" + " " * 30 + "\r", end="", flush=True)
                    print(f"Stell\t> ", end="", flush=True)
                    first_chunk = False
                
                content = msg.content
                # Handle Gemini's content format (list of dicts with 'text' key)
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and 'text' in block:
                            print(block['text'], end="", flush=True)
                else:
                    print(content, end="", flush=True)
        print("\n")