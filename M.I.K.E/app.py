import os
import subprocess
import shutil
import numpy as np
import faiss
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

# Ensure KB folder exists
os.makedirs("KB", exist_ok=True)

# Initialize embeddsing model and FAISS index
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dimension = 384
faiss_index_path = "faiss_index.idx"

# Load FAISS index if exists, otherwise create a new one
if os.path.exists(faiss_index_path):
    index = faiss.read_index(faiss_index_path)
    print(f"✅ Loaded FAISS index with {index.ntotal} embeddings.")
else:
    index = faiss.IndexFlatL2(dimension)
    print("🆕 Created a new FAISS index.")

# Define the retrieval-based prompt template
template = """ 
Use the information below to answer the question.

Context: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model='mistral')
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model 

def get_unique_filename(directory, filename):
    """Ensure unique filenames to avoid overwriting."""
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename

def call_pdf_extract(file_name):
    """Calls the PDF extraction script and reloads FAISS index after processing."""
    script_path = "pdfextrct.py"
    file_path = os.path.join("KB", file_name)

    if os.path.exists(file_path):
        print(f"📄 Calling PDF extraction for {file_name}...")
        subprocess.run(["python3", script_path, file_path])

        # Reload FAISS index after extraction
        global index
        if os.path.exists(faiss_index_path):
            index = faiss.read_index(faiss_index_path)
            print(f"✅ Reloaded FAISS index with {index.ntotal} embeddings.")
        else:
            print("❌ Failed to reload FAISS index.")
    else:
        print("❌ File not found in KB folder.")


def retrieve_context(query, top_k=3):
    """Retrieve the most relevant chunks from FAISS."""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    if indices[0][0] == -1:  # No valid results found
        print("⚠️ No relevant chunks found in FAISS.")
        return "No relevant context found."

    retrieved_chunks = []
    for idx in indices[0]:
        if idx != -1:
            chunk_path = f"split_chunks/chunk_{idx}.txt"
            if os.path.exists(chunk_path):
                with open(chunk_path, "r", encoding="utf-8") as file:
                    chunk_text = file.read()
                    retrieved_chunks.append(chunk_text)

    return "\n\n".join(retrieved_chunks)

def clear_kb():
    """Clears the KB and extracted files."""
    for folder in ["KB", "split_chunks", "data_embeddings"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"🗑️ Deleted {folder} and all extracted files.")
    os.makedirs("KB", exist_ok=True)
    global index
    index = faiss.IndexFlatL2(dimension)  # Reset FAISS index
    if os.path.exists(faiss_index_path):
        os.remove(faiss_index_path)

def handle_convo():
    """Handles the chatbot's conversation loop."""
    print("🤖 Welcome to AI ChatBot, Type 'exit' to quit")

    while True:
        user_input = input("Ask away: ")

        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        if user_input.lower() == "clear":
            clear_kb()
            print("🧹 Memory and all extracted files cleared.")
            continue

        if user_input.lower() == "upload":
            file_path = input("📂 Enter file path: ")
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                unique_file_name = get_unique_filename("KB", file_name)
                dest_path = os.path.join("KB", unique_file_name)
                shutil.copy(file_path, dest_path)
                print(f"✅ File uploaded successfully to KB/{unique_file_name}!")
                call_pdf_extract(unique_file_name)
            else:
                print("❌ File not found. Please try again.")
            continue

        # Retrieve relevant context from FAISS
        context = retrieve_context(user_input)

        # Generate response
        result = chain.invoke({"context": context, "question": user_input})
        print("🧠 AI: ", result)

        # Save FAISS index
        faiss.write_index(index, faiss_index_path)

if __name__ == "__main__":
    handle_convo()
