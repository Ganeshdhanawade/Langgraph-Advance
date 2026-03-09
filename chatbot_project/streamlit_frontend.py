import streamlit as st
from langraph_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage

CONFIG = {'configurable' : {'thread_id' : 'thread_1'}}

#st.session_state-not errase history_state
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


#previous history print
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('type here')

if user_input:

    st.session_state['message_history'].append({'role':'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    responce = chatbot.invoke({'messages' : [HumanMessage(content=user_input)]},config = CONFIG)
    AI_message = responce['messages'][-1].content
    st.session_state['message_history'].append({'role':'assistant', 'content': AI_message})
    with st.chat_message('assistant'):
        st.text(AI_message)