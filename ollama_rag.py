
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from torch import cuda
device = "cuda" if cuda.is_available() else "cpu"

st.set_page_config(page_title="Chatbot (Folder Version)")

class ChatPDF:
    def __init__(self):
        self.vector_db = None
        self.llm = Ollama(model="""hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:IQ4_XS""", base_url = "http://ollama:11434")
        self.chain = None
        self.processed_files = []  # Track processed files

    def ingest(self, file_path):
        self.processed_files.clear()
        # Load the PDF
        loader = PyMuPDFLoader(file_path=file_path)
        data = loader.load()
        st.info("PDF loaded successfully")

        # Split the PDF into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(data)
        st.info(f"Text split into {len(chunks)} chunks")

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


def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}..."):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    st.header("Chatbot!! - (PDF Version)")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

if __name__ == "__main__":
    page()
