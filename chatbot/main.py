import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import typing as t
import streamlit as st
from streamlit_feedback import streamlit_feedback
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langsmith import Client
from loguru import logger
from dotenv import load_dotenv
from chatbot import document_loader
import chromadb

from langchain.vectorstores import Chroma
import traceback
from chatbot import chain
from data_pipeline import etl
import os


# loading environment variables
load_dotenv()

streamlit_memory = StreamlitChatMessageHistory()

# persisting chat history to memory
memory = ConversationBufferMemory(
    chat_memory=streamlit_memory,
    return_messages=True,
    memory_key='chat_history'
)

# initialize chroma client
client  = chromadb.Client()

# create the embedding object
embeddings = OllamaEmbeddings(model='deepseek-r1')
 # retrieve existing collection
COLLECTION_NAME ='amazon_collection'

# to clear message history from the UI
if st.sidebar.button(" Clear message history"):
    logger.info('clearing message history...')
    memory.clear()
    st.session_state.trace_link = None
    st.session_state.run_id = None
    st.session_state.clear()

def user_bot_system(amazon_retriever:t.Optional[t.Any | None] = None, 
                    db:t.Optional[t.Any | None] = None):
    """user bot function"""

    # initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # user input definition
    if prompt := st.chat_input('Ask your question: '):
        # display user message in chat container
        with st.chat_message('user'):
            st.markdown(prompt)
        # adding user message to chat history
        st.session_state.messages.append({'role': 'user', 'content':prompt})

        if db:
            try:
                db_result = db.similarity_search(query=prompt)
            except Exception as e:
                logger.error(f'Caught query error: {e}')
                traceback.print_exc()
            else:
                formatted_result = chain.combine_docs(docs=db_result)
                bot_response = chain.chat_model(prompt, formatted_result)

                # display assitant response in chat message container
                with st.chat_message('assistant'):
                    st.markdown(bot_response)

                # adding assistance response to chat history
                st.session_state.messages.append({'role':'assistant', 'content': bot_response})

        else:
            bot_response = chain.rag_chain(question=prompt, _retriever=amazon_retriever)

            # display assitant response in chat message container
            with st.chat_message('assistant'):
                # response = st.write_stream(bot_response)
                st.markdown(bot_response)
                
            # adding assistance response to chat history
            st.session_state.messages.append({'role':'assistant', 'content': bot_response})

# check if vectorstore exists already, if not load data from api, process this data and save in the mysql database
if __name__ == "__main__":

    if os.path.exists('./chatbot/chromadb'): # path to the persisted collection
        logger.info('vector store directory exits, Using it...')

        # load a chroma collection
        db_collection = Chroma(
                    collection_name= 'amazon_collection',
                    persist_directory='./chatbot/chromadb/',
                    embedding_function=embeddings
                    )

        user_bot_system(db=db_collection)
        
    else:
        # activity 1
        logger.info('vector store does not exists creating new one....')
        # etl.etl() # uncomment if no data exists in the mysql database

        # activity 2
        # load data from database, convert to documents and then subsequent embeddings and save to a chroma vectostore
        documents = document_loader.document_loader()
        retriever = document_loader.document_embeddings(documents)

        user_bot_system(amazon_retriever=retriever)
