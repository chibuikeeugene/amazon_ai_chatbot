import streamlit as st
from streamlit_feedback import streamlit_feedback
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_ollama.embeddings import OllamaEmbeddings
from langsmith import Client
from loguru import logger
from dotenv import load_dotenv
from rag_chain import document_loader, chain
from data_pipeline import etl
import os
import chromadb
from langchain.vectorstores import Chroma



# loading environment variables
load_dotenv()

streamlit_memory = StreamlitChatMessageHistory()

# persisting chat history to memory
memory = ConversationBufferMemory(
    chat_memory=streamlit_memory,
    return_messages=True,
    memory_key='history'
)


# to clear message history from the UI
if st.sidebar.button(" Clear message history"):
    logger.info('clearing message history...')
    memory.clear()
    # st.session_state.trace_link = None
    st.session_state.run_id = None
    st.session_state.clear()


# ================= chat user interface =============== #
st.title('Amazon customer chatbot')

# ============== bot logic =============== #
def bot_system(db_store, mem, prompts):
    """ this function creates the retrieval chain, handles the bot's contextual response and saves response to memory"""

    global streamlit_memory

    # instantiate the retrieval chain
    conversation_retriever_chain = chain.retrieval_chain(db_store, mem)

     # retrieve the response 
    response = conversation_retriever_chain.invoke(
        {'question': prompts, 'chat_history': }
    )

    # display assitant response in chat message container
    with st.chat_message('assistant'):
        st.markdown(response)

    # adding assistance response to chat history
    st.session_state.messages.append({'role':'assistant', 'content': response})


# check if vectorstore exists already, if not load data from api, process this data and save in the mysql database
if __name__ == "__main__":

    if os.path.exists('./rag_chain/'): # path to the saved vectorstore
        # initialize chroma client
        client  = chromadb.Client()

        # create the embedding object
        embeddings = OllamaEmbeddings(model='llama3.1')

        # load the vectorstore from directory
        vectorstore = Chroma(
            client=client,
            persist_directory='./rag_chain/',
            embedding_function= embeddings
        )
    else:
        # activity 1
        etl.etl()

        # activity 2
        # load data from database, convert to documents and then subsequent embeddings and save to a chroma vectostore
        documents = document_loader.document_loader()
        vectorstore = document_loader.document_embeddings(documents)
    
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

    bot_system(vectorstore, memory, prompt)