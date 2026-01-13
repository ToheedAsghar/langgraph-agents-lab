import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage

# session state to store messages
    
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

CONFIG = {
    'configurable': {
        'thread_id': '123'
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

    res = chatbot.invoke({
        'messages': [HumanMessage(content=user_input)]
    }, config=CONFIG)

    st.session_state['messages'].append({"role": "assistant", "content": res['messages'][-1].content})
    with st.chat_message("assistant"): # role is assistant
        st.text(res['messages'][-1].content)