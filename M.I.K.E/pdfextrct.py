import os
import fitz  # PyMuPDF
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import sys

# Define folders
kb_folder = "KB"
chunk_folder = "split_chunks"
embedding_folder = "data_embeddings"

for folder in [kb_folder, chunk_folder, embedding_folder]:
    os.makedirs(folder, exist_ok=True)

# Initialize embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dimension = 384
faiss_index_path = "faiss_index.idx"

# Load or initialize FAISS index
if os.path.exists(faiss_index_path):
    index = faiss.read_index(faiss_index_path)
    print(f"✅ Loaded FAISS index with {index.ntotal} embeddings.")
else:
    index = faiss.IndexFlatL2(dimension)
    print("🆕 Created a new FAISS index.")

def sanitize_filename(filename):
    """Sanitize filename to prevent issues."""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def extract_text(file_path):
    """Extracts text from a PDF using pdfplumber."""
    structured_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                structured_text.append(text)

    full_text = "\n\n".join(structured_text)
    return full_text

def split_text(text):
    """Splits text into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def process_pdf(file_path):
    """Processes a PDF file and stores embeddings in FAISS."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📄 Processing {file_path}...")

    text = extract_text(file_path)
    chunks = split_text(text)

    print(f"📝 Extracted {len(chunks)} chunks.")

    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

    print(f"📏 Embeddings shape: {embeddings.shape}")

    if embeddings.shape[1] != dimension:
        print(f"⚠️ Dimension mismatch! Expected: {dimension}, Got: {embeddings.shape[1]}")
        return

    for i, chunk in enumerate(chunks):
        chunk_filename = f"chunk_{index.ntotal + i}.txt"
        chunk_path = os.path.join(chunk_folder, chunk_filename)
        with open(chunk_path, "w", encoding="utf-8") as file:
            file.write(chunk)
        print(f"💾 Saved chunk: {chunk_path}")

    index.add(embeddings)
    faiss.write_index(index, faiss_index_path)

    print(f"✅ Stored {len(chunks)} chunks in FAISS.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_pdf(sys.argv[1])
    else:
        print("❌ No PDF file provided.")
