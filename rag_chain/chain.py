from langchain_ollama.chat_models import ChatOllama
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import streamlit as st

@st.cache_resource(show_spinner=False)
def retrieval_chain(_vectore_store, _memory):
    """ Returns a conversational retrieval chain with memory
    args:
    * _vectore_store - an instance of the vectorstore

    * _memory - to persist chat history interactions

    return:
    * conversational retrieval chain
    """
    # creating an instance of the chatmodel
    chat_model = ChatOllama(
       model='llama3.1',
       num_predict= 1000,
       temperature=0,
       
    )

    # load retriever object
    retriever = _vectore_store.as_retriever()

    # creating the chat retriever chain object
    chat_retreiver_chain  = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=retriever,
        memory = _memory,
    )

    return chat_retreiver_chain
    