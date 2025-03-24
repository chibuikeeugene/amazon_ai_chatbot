import streamlit as st
from streamlit_feedback import streamlit_feedback
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langsmith import Client
from loguru import logger
from rag_chain import document_loader, chain, prompt_template
from dotenv import load_dotenv


# loading environment variables
load_dotenv()

# persisting chat history to memory
memory = ConversationBufferMemory(
    chat_memory=StreamlitChatMessageHistory(),
    return_messages=True,
    memory_key='history'
)


# to clear message history from the UI
if st.sidebar.button(" Clear message history"):
    logger.info('clearing message history...')
    memory.clear()
    # st.session_state.trace_link = None
    # st.session_state.run_id = None
    st.session_state.clear()


