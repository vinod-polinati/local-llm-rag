import os
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer  # Use Sentence-Transformers model
import faiss  # FAISS for vector storage
import numpy as np
import re

# Define folders
kb_folder = "KB"
text_folder = "extracted_text"
image_folder = "extracted_images"
chunk_folder = "split_chunks"
embedding_folder = "data_embeddings"

# Ensure required folders exist
for folder in [kb_folder, text_folder, image_folder, chunk_folder, embedding_folder]:
    os.makedirs(folder, exist_ok=True)

# Initialize embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FAISS index
dimension = 384  # Dimensionality of all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)

def sanitize_filename(filename):
    """Sanitize the filename to remove special characters."""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def extract_text_and_tables(file_path):
    """Extract text and tables from a PDF while maintaining their relationship."""
    try:
        structured_text = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    structured_text.append(page_text)
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        table_text = "\n".join([" | ".join([str(cell) if cell is not None else "" for cell in row]) for row in table])
                        structured_text.append(f"[Table from Page {page_num + 1}]\n{table_text}\n")
        
        full_text = "\n\n".join(structured_text)
        sanitized_name = sanitize_filename(os.path.basename(file_path))
        text_filename = os.path.join(text_folder, f"{sanitized_name}.txt")
        with open(text_filename, "w", encoding="utf-8") as text_file:
            text_file.write(full_text)
        return full_text
    except Exception as e:
        print(f"Error extracting text and tables from {file_path}: {e}")
        return ""

def extract_images(file_path):
    """Extract images from a PDF."""
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    img = Image.open(BytesIO(image_bytes))
                    sanitized_name = sanitize_filename(os.path.basename(file_path))
                    img.save(
                        os.path.join(image_folder, f"{sanitized_name}_page{page_num + 1}_img{img_index + 1}.png"),
                        "PNG"
                    )
                except Exception as e:
                    print(f"Error saving image from {file_path}, page {page_num + 1}: {e}")
    except Exception as e:
        print(f"Error extracting images from {file_path}: {e}")

def split_text_recursive(text, chunk_size=500, chunk_overlap=50):
    """Split text using RecursiveCharacterTextSplitter while keeping structure."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def store_embeddings(file_name, embeddings):
    """Store embeddings in a file."""
    sanitized_name = sanitize_filename(file_name)
    embedding_file = os.path.join(embedding_folder, f"{sanitized_name}_embeddings.npy")
    np.save(embedding_file, embeddings)
    return embedding_file

def process_pdf(file_path):
    """Extract text, tables, images, split into structured chunks, store embeddings, and save in FAISS."""
    if os.path.exists(file_path):
        print(f"Processing {file_path}...")
        structured_text = extract_text_and_tables(file_path)
        extract_images(file_path)
        
        # Split text while keeping tables in context
        structured_chunks = split_text_recursive(structured_text)
        
        # Save chunked text files
        sanitized_name = sanitize_filename(os.path.basename(file_path))
        chunk_file = os.path.join(chunk_folder, f"{sanitized_name}_chunks.txt")
        with open(chunk_file, "w", encoding="utf-8") as file:
            for i, chunk in enumerate(structured_chunks):
                file.write(f"Chunk {i + 1}:\n{chunk}\n\n")
        
        # Generate embeddings for structured chunks
        chunk_embeddings = embedding_model.encode(structured_chunks, convert_to_numpy=True)
        
        # Store embeddings
        embedding_file = store_embeddings(os.path.basename(file_path), chunk_embeddings)
        print(f"Embeddings saved to {embedding_file}")
        
        # Store in FAISS
        global index
        index.add(chunk_embeddings)
        print(f"Stored {len(structured_chunks)} structured chunks in FAISS.")
    else:
        print(f"File not found: {file_path}")

def process_all_pdfs():
    """Process all PDFs in the KB folder."""
    files = [f for f in os.listdir(kb_folder) if f.lower().endswith(".pdf")]
    if not files:
        print("No PDF files found in KB folder.")
        return
    for file_name in files:
        file_path = os.path.join(kb_folder, file_name)
        process_pdf(file_path)

if __name__ == "__main__":
    process_all_pdfs()
