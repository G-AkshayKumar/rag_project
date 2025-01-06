import os
import tempfile
import streamlit as st
from streamlit_chat import message
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.ollama import Ollama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from torch import cuda

device = "cuda" if cuda.is_available() else "cpu"

st.set_page_config(page_title="Chatbot (Folder Version)")

class ChatPDF:
    def __init__(self):
        self.vector_db = None
        self.llm = Ollama(model="""hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:IQ4_XS""")
        self.chain = None
        self.processed_files = []  # Track processed files

    def ingest(self, folder_path):
        self.processed_files.clear()
      
        loader = PyPDFDirectoryLoader(folder_path)
        data = loader.load()
        st.info(f"PDFs loaded from directory successfully: {folder_path}")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(data)

        # Create vector database
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings(),
            collection_metadata={'hnsw':"cosine"},
            persist_directory='chromadb'
        )
        st.info("Vector database created successfully.")

        retriever2 = self.vector_db.as_retriever()
        keyword_retriever = BM25Retriever.from_documents(documents=chunks)
        main_retriever = EnsembleRetriever(retrievers = [retriever2, keyword_retriever], weights=[0.4,0.6])

        chat_prompt = ChatPromptTemplate.from_template(
            """
            You are an expert assistant designed to answer questions based on the provided information.
            Use the context below to respond accurately and concisely to the query.
            If the context does not contain the necessary information, state, 'The provided context does not contain enough information to answer the question'.

            Context:
            {context}

            Answer the question based on the above context:
            Question: {question}

            """
        )

        self.chain = (
            {"context": main_retriever, "question": RunnablePassthrough()}
            | chat_prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question):
        if not self.chain:
            return "Please provide a valid folder path and ingest files first."
        return self.chain.invoke(question)

    def clear(self):
        self.vector_db = None
        self.chain = None
        self.processed_files.clear()


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"].strip():
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))
        st.session_state["user_input"] = ""


def read_and_ingest_folder():
    folder_path = st.session_state["folder_path"].strip()
    if os.path.isdir(folder_path):
        st.session_state["assistant"].clear()
        st.session_state["messages"] = []
        st.session_state["user_input"] = ""
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting files from {folder_path}..."):
            st.session_state["assistant"].ingest(folder_path)
    else:
        st.error("Invalid folder path. Please enter a valid directory.")


def page():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    st.header("ChatBot (Folder Version)")

    st.subheader("Enter Folder Path")
    st.text_input(
        "Folder Path",
        key="folder_path",
        on_change=read_and_ingest_folder
    )

    st.session_state["ingestion_spinner"] = st.empty()
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
