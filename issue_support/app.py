### LEGACY IMPLEMENTATION OF TESTCASE UI; NEED TO UPDATE WHEN BACKEND IS COMPLETED

import config.env_setup
import asyncio

# Ensure an asyncio event loop exists in this thread (Streamlit's script thread)
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from graph.workflow import app
import streamlit as st
from utils.helpers import output_formatter

st.title("Issue Support")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_question := st.chat_input("How can I help you?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
    print("session state:",st.session_state.messages)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Invoke LangGraph app with user question
        response_from_langgraph = output_formatter(app.invoke({"messages":[user_question]}))
        # Show response in UI
        st.write(response_from_langgraph)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_from_langgraph})