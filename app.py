import streamlit as st
from langchain.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# === Settings ===
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_PATH = "chroma_db"
DOC_PATH = "engineer.txt"

# === Initialize session ===
st.set_page_config(page_title="🧠 Local RAG Chatbot", layout="wide")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Load and process documents ===
@st.cache_resource
def setup_vectordb():
    loader = TextLoader(DOC_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectordb = Chroma.from_documents(split_docs, embedding_model=embeddings, persist_directory=CHROMA_PATH)
    vectordb.persist()
    return vectordb

# === Setup QA Chain ===
@st.cache_resource
def setup_chain():
    vectordb = setup_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="mistral")  # or llama2, gemma, etc.
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, return_source_documents=True)
    return chain

qa_chain = setup_chain()

# === UI ===
st.title("📚 Local RAG Chatbot (Ollama + ChromaDB)")
st.markdown("Ask the chatbot!")

# Chat input
user_input = st.text_input("You:", placeholder="Ask me anything about the document...")

# Handle query
if user_input:
    result = qa_chain.invoke({"question": user_input})
    response = result["answer"]
    st.session_state.chat_history.append((user_input, response))

# Display chat history
for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")