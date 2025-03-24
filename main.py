import streamlit as st
from streamlit_feedback import streamlit_feedback
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langsmith import Client
from loguru import logger
from dotenv import load_dotenv
from rag_chain import document_loader, chain
from data_pipeline import etl



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


# ================= chat user interface =============== #
st.title('Amazon customer chatbot')

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

# ============== bot logic =============== #
def bot_system(db_store, mem):
    """ this function creates the retrieval chain, handles the bot's contextual response and saves response to memory"""
    # instantiate the retrieval chain
    conversation_retriever_chain = chain.retrieval_chain(db_store, mem)

    # display assitant response in chat message container
    with st.chat_message('assistant'):
        response = st.write(conversation_retriever_chain.stream(prompt))

    # adding assistance response to chat history
    st.session_state.messages.append({'role':'assistant', 'content': response})


# load data from api, process this data and save in the mysql database
if __name__ == "__main__":
    # activity 1
    etl.etl()

    # activity 2
    # load data from database, convert to documents and then subsequent embeddings and save to a chroma vectostore
    documents = document_loader.document_loader()
    vectorstore = document_loader.document_embeddings(documents)

    bot_system(vectorstore, memory)