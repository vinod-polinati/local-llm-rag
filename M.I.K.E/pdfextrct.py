import os
import fitz  # PyMuPDF
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# Define folders
kb_folder = "KB"
text_folder = "extracted_text"
chunk_folder = "split_chunks"
embedding_folder = "data_embeddings"

for folder in [kb_folder, text_folder, chunk_folder, embedding_folder]:
    os.makedirs(folder, exist_ok=True)

# Initialize embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dimension = 384
faiss_index_path = "faiss_index.idx"

# Load FAISS index
if os.path.exists(faiss_index_path):
    index = faiss.read_index(faiss_index_path)
else:
    index = faiss.IndexFlatL2(dimension)

def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def extract_text(file_path):
    structured_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                structured_text.append(text)
    
    full_text = "\n\n".join(structured_text)
    sanitized_name = sanitize_filename(os.path.basename(file_path))
    text_filename = os.path.join(text_folder, f"{sanitized_name}.txt")
    
    with open(text_filename, "w", encoding="utf-8") as text_file:
        text_file.write(full_text)
    
    return full_text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def process_pdf(file_path):
    if os.path.exists(file_path):
        print(f"Processing {file_path}...")
        
        text = extract_text(file_path)
        chunks = split_text(text)
        
        sanitized_name = sanitize_filename(os.path.basename(file_path))
        
        embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
        
        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(chunk_folder, f"chunk_{len(index)+i}.txt")
            with open(chunk_path, "w", encoding="utf-8") as file:
                file.write(chunk)
        
        index.add(embeddings)
        faiss.write_index(index, faiss_index_path)
        print(f"Stored {len(chunks)} chunks in FAISS.")
    else:
        print(f"File not found: {file_path}")

if __name__ == "__main__":
    for file in os.listdir(kb_folder):
        if file.endswith(".pdf"):
            process_pdf(os.path.join(kb_folder, file))
