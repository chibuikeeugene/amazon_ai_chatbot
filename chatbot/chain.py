from langchain.schema import Document
import ollama
import re
import streamlit as st
from loguru import logger


# combine document function
def combine_docs(docs: list[Document]):
    """this function merges documents into a single string.
    this is necessary to format our inputs into a form understandable and processable by
    our model
    
    args:
    * doc: list[Document]
    
    """
    return '\n\n'.join(doc.page_content for doc in docs)

def chat_model(question:str, context:str):
    """ function formats the user’s question and the retrieved document context into a structured prompt. 
    
    args:
    * question:str - query
    * context:str - context from the retrieved document returned from the retriever
    
    """

    formatted_prompt = f'Question: {question}\n\nContext:{context}'

    # This formatted input is then sent to DeepSeek-R1 via ollama.chat(), 
    # which processes the question within the given context and returns a relevant answer.

    final_response = ollama.chat(
        model='deepseek-r1',
        messages=[{"role": "user", "content": formatted_prompt}],
        stream=False
    )
    response = final_response['message']['content']
    final_answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return final_answer

@st.cache_resource(show_spinner=False)
# defining the rag chain function
def rag_chain(question, _retriever):
    retrieved_docs = _retriever.invoke(question)
    formatted_content = combine_docs(retrieved_docs)
    return chat_model(question, formatted_content)