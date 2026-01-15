import streamlit as st
import uuid # to create unique thread ids
from chatbot_backend import chatbot, generate_thread_topic
from langchain_core.messages import HumanMessage

### -- utility functions --- ###

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['messages'] = []
    st.session_state['current_topic'] = None
    
def add_thread(thread_id, thread_name):
    if thread_id not in [t[0] for t in st.session_state['chat_threads']]:
        st.session_state['chat_threads'].append((thread_id, thread_name))

def load_thread(thread_id):
    return chatbot.get_state(config={
        'configurable': {'thread_id': thread_id}
    }).values['messages']

### -- session state to store messages -- ###

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'current_topic' not in st.session_state:
    st.session_state['current_topic'] = None

if 'thread_topics' not in st.session_state:
    st.session_state['thread_topics'] = {}  # maps thread_id -> topic

### -- SIDEBAR UI -- ###

st.sidebar.title("Chat-LLM")
st.sidebar.button("New Chat", on_click=reset_chat)
st.sidebar.header("Conversations")
for thread_id in st.session_state['chat_threads'][::-1]:
    # thread_id? name : id
    topic = st.session_state['thread_topics'].get(thread_id, f"Thread: {thread_id[:8]}")
    if st.sidebar.button(topic, key=thread_id):
        msgs = load_thread(thread_id)
        st.session_state['thread_id'] = thread_id
        st.session_state['current_topic'] = st.session_state['thread_topics'].get(thread_id)

        tmp_msgs = []
        for msg in msgs:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            tmp_msgs.append({"role": role, "content": msg.content})
        st.session_state['messages'] = tmp_msgs

CONFIG = {
    'configurable': {
        'thread_id': st.session_state['thread_id']
    }
}

for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    is_first_message: bool = len(st.session_state['messages']) == 0
    st.session_state['messages'].append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.text(user_input)

    if is_first_message:
        topic = generate_thread_topic(user_input)
        st.session_state['current_topic'] = topic
        st.session_state['thread_topics'][st.session_state['thread_id']] = topic
        if st.session_state['thread_id'] not in st.session_state['chat_threads']:
            st.session_state['chat_threads'].append(st.session_state['thread_id'])

    with st.chat_message("assistant"):
        with st.status("Thinking ...", expanded=True) as status:
            res_chunks = []
            stream = chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                stream_mode='messages',
                config=CONFIG
                )

            first_chunk, first_meta = next(stream)
            res_chunks.append(first_chunk)
            status.update(label="Responding ...", state='complete', expanded=False)
        
        def stream_response():
            for content in res_chunks:
                yield content

            for chunk, meta in stream:
                yield chunk.content

        ai_msg = st.write_stream(stream_response())


    st.session_state['messages'].append(
        {"role": "assistant", "content": ai_msg}
    )