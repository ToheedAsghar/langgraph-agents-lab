import streamlit as st
import uuid # to create unique thread ids
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage

### -- utility functions --- ###

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['chat_threads'].append(thread_id)
    st.session_state['messages'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_thread(thread_id):
    return chatbot.get_state(config={
        'configurable': {'thread_id': thread_id}
    }).values['messages']

# -- session state to store messages

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

## -- SIDEBAR UI -- #

st.sidebar.title("Chat-LLM")
st.sidebar.button("New Chat", on_click=reset_chat)
st.sidebar.header("New Conversation")
for id in st.session_state['chat_threads'][::-1]:
    # st.sidebar.button(f"Thread: {id[:8]}", on_click=load_thread, args=(id))
    if st.sidebar.button(id):
        msgs = load_thread(id)
        st.session_state['thread_id'] = id

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

# loading the conversation history
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type your message here...")

if user_input:

    st.session_state['messages'].append({"role": "user", "content": user_input})
    
    with st.chat_message("user"): # role is user
        st.text(user_input)

    # res = chatbot.invoke({
    #     'messages': [HumanMessage(content=user_input)]
    # }, config=CONFIG)

    #st.session_state['messages'].append({"role": "assistant", "content": res['messages'][-1].content})
    with st.chat_message("assistant"): # role is assistant
        ai_msg = st.write_stream(
            chunk.content for chunk, meta_data in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            stream_mode='messages',
            config=CONFIG
            )
        )

    st.session_state['messages'].append(
        {"role": "assistant", "content": ai_msg}
    )