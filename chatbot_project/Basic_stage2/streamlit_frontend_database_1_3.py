import streamlit as st
from langraph_database_backend1 import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid  #support for randomly gerate the thread id

#************************* utility functions *******************
def genrate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = genrate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_converstion(thread_id):
    state =  chatbot.get_state(config ={'configurable' : {'thread_id' : thread_id}})
    #check if messages key exist in the state values, return impty list if not
    return state.values.get('messages',[])


#****************** Session setup ********************************
#st.session_state-not errase history_state
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = genrate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])


# ****************config the tread ************************
CONFIG = {'configurable' : {'thread_id' : st.session_state['thread_id']}}


#******************* Sidebar UI ********************
st.sidebar.title('Langgrpah chatbot')

if st.sidebar.button('new_chat'):
    reset_chat()

st.sidebar.header('My conversation')

for thread_id in st.session_state['chat_threads']:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id']= thread_id
        messages = load_converstion(thread_id)
        
        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role':role,'content':msg.content})

        st.session_state['message_history'] = temp_messages


#**************************** Main UI *********************************
#previous history print
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('type here')

if user_input:

    st.session_state['message_history'].append({'role':'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)


    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())
        print(ai_message)

    st.session_state['message_history'].append({'role':'assistant', 'content': ai_message})
