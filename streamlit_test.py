import os
import tempfile
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from torch import cuda

device = "cuda" if cuda.is_available() else "cpu"

st.set_page_config(page_title="Chatbot (Folder Version)")

class ChatPDF:
    def __init__(self):
        self.vector_db = None
        self.llm = ChatOllama(model="llama3.1")
        self.chain = None
        self.processed_files = []  # Track processed files

    def ingest(self, folder_path):
        self.processed_files.clear()
        all_files = [
            os.path.join(folder_path, file) 
            for file in os.listdir(folder_path) if file.endswith(".pdf")
        ]

        if not all_files:
            st.error("No PDF files found in the specified folder.")
            return
        
        st.info(f"Found {len(all_files)} PDFs in the folder.")

        # Load and process all files
        all_chunks = []
        for file_path in all_files:
            loader = PyMuPDFLoader(file_path=file_path)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            all_chunks.extend(chunks)
            self.processed_files.append(file_path)
            st.info(f"Processed: {file_path} ({len(chunks)} chunks)")

        # Create vector database
        self.vector_db = Chroma.from_documents(
            documents=all_chunks,
            embedding=FastEmbedEmbeddings(),
            collection_name="local-rag",
            persist_directory='chroma_db'
        )
        st.info("Vector database created successfully.")

        # Print all ingested files
        st.subheader("Ingested Files:")
        for file in self.processed_files:
            st.write(f" {file}")

        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Generate 2
            different versions of the given user question to retrieve relevant documents from
            a vector database. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

        retriever = MultiQueryRetriever.from_llm(
            self.vector_db.as_retriever(),
            self.llm,
            prompt=query_prompt,
        )

        chat_prompt = ChatPromptTemplate.from_template(
            """
            You are an expert assistant with access to the following context extracted from documents. Your job is to answer the user's question as accurately as possible, using the context below.

            Context:
            {context}

            Given this information, please provide a comprehensive and relevant answer to the following question:
            Question: {question}

            If the context does not contain enough information, clearly state that the information is not available in the context provided.
            """
        )

        self.chain = (
            {"context": retriever, "question": RunnablePassthrough()}
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
