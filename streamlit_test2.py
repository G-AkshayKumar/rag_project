import streamlit as st
from langchain import retrievers
from langchain_community.document_loaders import UnstructuredPDFLoader
#from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
#from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.llms import HuggingFaceEndpoint
#from langchain_huggingface.llms import HuggingFacePipeline
#from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForQuestionAnswering, pipeline
#from llama_cpp import Llama




import warnings
warnings.filterwarnings('ignore')


# Set environment variable for protobuf
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

local_path = r"C:\AKSHAY\ACAD FOLDER\SEM4\datafile\b.tech-b.tech-hons-ordinances-and-regulations-2010-14.pdf"
folder_path = r"C:\AKSHAY\ACAD FOLDER\SEM4\datafile"

"""
if local_path:
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
    print(f"PDF loaded successfully: {local_path}")
else:
    print("Upload a PDF file")
"""

loader = PyPDFDirectoryLoader(folder_path)
data = loader.load()
print(f"PDF loaded successfully: {folder_path}")

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(data)
print(f"Text split into {len(chunks)} chunks")
#print(chunks)

# Create vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=FastEmbedEmbeddings(),
    collection_metadata={'hnsw':"cosine"},
    persist_directory='chroma_db2'
)
print("Vector database created successfully")


"""
db2 = Chroma(embedding_function=OllamaEmbeddings(model='nomic-embed-text'))
results = db2.similarity_search_with_score(chunks, k=5)

context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
"""


local_model = "mistral" 
local_model2 = "llama3.1"
local_model3 = """hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:IQ4_XS"""
llm = Ollama(model=local_model3, base_url = "http://localhost:11434")


repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
repo_id2 = "bartowski/Llama-3.2-3B-Instruct-GGUF"

llm2 = HuggingFaceEndpoint(
    repo_id=repo_id2, max_length=128, temperature=0.5, token="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"
)

"""

model_id = "tiiuae/falcon-7b-instruct"
model_id2 ="deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_id2)
model = AutoModelForCausalLM.from_pretrained(model_id2)
model2 = AutoModelForQuestionAnswering.from_pretrained(model_id2)
#pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
pipe2 = pipeline("question-answering",model=model2, tokenizer=tokenizer)
llm3 = HuggingFacePipeline(pipeline=pipe2)
"""


QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate 2
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Set up retriever
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(search_kwargs={'k': 10}), 
    llm,
    prompt=QUERY_PROMPT
)

retriever2 = vector_db.as_retriever()
keyword_retriever = BM25Retriever.from_documents(documents=chunks)
main_retriever = EnsembleRetriever(retrievers = [retriever2, keyword_retriever], weights=[0.4,0.6])
"""
compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
"""
template = """
You are an expert assistant designed to answer questions based on the provided information.
Use the context below to respond accurately and concisely to the query.
If the context does not contain the necessary information, state, 'The provided context does not contain enough information to answer the question'.
{context}

---

Answer the question based on the above context: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
# Create chain
chain = (
    {"context": main_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


question = """What is the company’s policy on stock trading for employees?"""
print(chain.invoke(question))





