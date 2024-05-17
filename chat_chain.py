from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory

from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st

from retriever import *


def build_document_chain(llm):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a chatbot, which chat with users based on the chat history between you and also answer the user's questions based on the below context:\n\n{context}.If you don't know the answer, please let the user know."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain

def build_chat_chain(context_dir):

    chat_history_record = st.session_state.chat_history_record

    llm = ChatOpenAI()

    vector = vector_store(context_dir)
    retriever = vector.as_retriever()

    retriever_chain = build_retriever_chain(llm, retriever)

    document_chain = build_document_chain(llm)

    chain_with_message_history = RunnableWithMessageHistory(
        document_chain,
        lambda session_id: chat_history_record,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    chat_chain = create_retrieval_chain(retriever_chain, chain_with_message_history)

    return chat_chain

