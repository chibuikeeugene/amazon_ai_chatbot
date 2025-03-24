from rag_chain.document_loader import document_loader 
from rag_chain.document_loader import document_embeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.llms import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import streamlit as st

@st.cache_resource(show_spinner=False)
def retrieval_chain(vectore_store, system_prompt, _memory):
    """ Returns a conversational retrieval chain with memory
    args:
    vectore_store - an instance of the vectorstore

    prompt-template - 
    """
    # creating an instance of the chatmodel
    chat_model = ChatOllama(
       model='llama3.1',
       num_predict= 1000,
       temperature=0,
       
    )

    # load retriever object
    retriever = vectore_store.retriever()

    # creating the chat retriever chain object
    chat_retreiver_chain  = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=retriever,
        memory = _memory,
        condense_question_prompt = system_prompt
    )

    return chat_retreiver_chain
    