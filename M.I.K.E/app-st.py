import os
import shutil
import faiss
import numpy as np
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
import subprocess

# Constants
dimension = 384
faiss_index_path = "faiss_index.idx"
os.makedirs("KB", exist_ok=True)

# Initialize embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load or create FAISS index
if os.path.exists(faiss_index_path):
    index = faiss.read_index(faiss_index_path)
else:
    index = faiss.IndexFlatL2(dimension)

# LLM and prompt setup
template = """
Use the information below to answer the question.

Context: {context}

Question: {question}

Answer:
"""
model = OllamaLLM(model='mistral')
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Helper Functions
def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename

def call_pdf_extract(file_name):
    script_path = "pdfextrct.py"
    file_path = os.path.join("KB", file_name)
    if os.path.exists(file_path):
        subprocess.run(["python3", script_path, file_path])
        global index
        if os.path.exists(faiss_index_path):
            index = faiss.read_index(faiss_index_path)

def retrieve_context(query, top_k=3):
    if index.ntotal == 0:
        return "FAISS index is empty. Upload a document first."
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    if indices[0][0] == -1:
        return "No relevant context found."

    retrieved_chunks = []
    for idx in indices[0]:
        if idx != -1:
            chunk_path = f"split_chunks/chunk_{idx}.txt"
            if os.path.exists(chunk_path):
                with open(chunk_path, "r", encoding="utf-8") as file:
                    retrieved_chunks.append(file.read())
    return "\n\n".join(retrieved_chunks)

def clear_kb():
    for folder in ["KB", "split_chunks", "data_embeddings"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    os.makedirs("KB", exist_ok=True)
    global index
    index = faiss.IndexFlatL2(dimension)
    if os.path.exists(faiss_index_path):
        os.remove(faiss_index_path)

# Streamlit UI
st.set_page_config(page_title="📚 AI PDF Chatbot", layout="centered")
st.title("- M.I.K.E -")

with st.sidebar:
    st.header("📂 Upload and Manage")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        file_name = get_unique_filename("KB", uploaded_file.name)
        file_path = os.path.join("KB", file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Uploaded {file_name}")
        call_pdf_extract(file_name)

    if st.button("🧹 Clear All Data"):
        clear_kb()
        st.success("Memory and all extracted files cleared.")

    st.write("📄 Uploaded Files:")
    for file in os.listdir("KB"):
        st.markdown(f"- {file}")

st.divider()

user_question = st.text_input("Ask away:")

if user_question:
    with st.spinner("Retrieving answer..."):
        context = retrieve_context(user_question)
        result = chain.invoke({"context": context, "question": user_question})
        st.success("Answer:")
        st.write(result)

# Save FAISS index after each request
faiss.write_index(index, faiss_index_path)
