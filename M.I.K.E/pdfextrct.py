import os
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer  # Use Sentence-Transformers model
import faiss  # FAISS for vector storage
import numpy as np

# Define folders
kb_folder = "KB"
text_folder = "extracted_text"
table_folder = "extracted_tables"
image_folder = "extracted_images"
chunk_folder = "split_chunks"
embedding_folder = "data_embeddings"

# Ensure required folders exist
for folder in [kb_folder, text_folder, table_folder, image_folder, chunk_folder, embedding_folder]:
    os.makedirs(folder, exist_ok=True)

# Initialize embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FAISS index
dimension = 384  # Dimensionality of all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)

def extract_text(file_path):
    """Extract text from a PDF and save it."""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        text_filename = os.path.join(text_folder, os.path.basename(file_path) + ".txt")
        with open(text_filename, "w", encoding="utf-8") as text_file:
            text_file.write(text)
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def extract_tables(file_path):
    """Extract tables from a PDF and save them."""
    try:
        table_texts = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        table_text = "\n".join([" | ".join([str(cell) if cell is not None else "" for cell in row]) for row in table])
                        table_texts.append(f"Table from Page {page_num+1}:\n{table_text}\n")
        table_filename = os.path.join(table_folder, os.path.basename(file_path) + ".txt")
        with open(table_filename, "w", encoding="utf-8") as table_file:
            table_file.write("\n".join(table_texts))
        return "\n".join(table_texts)
    except Exception as e:
        print(f"Error extracting tables: {e}")
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
                img = Image.open(BytesIO(image_bytes))
                img.save(os.path.join(image_folder, f"{os.path.basename(file_path)}_page{page_num+1}_img{img_index+1}.png"), "PNG")
    except Exception as e:
        print(f"Error extracting images: {e}")

def split_text_recursive(text, chunk_size=500, chunk_overlap=50):
    """Split text using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def store_embeddings(file_name, embeddings):
    """Store embeddings in a file."""
    embedding_file = os.path.join(embedding_folder, f"{file_name}_embeddings.npy")
    np.save(embedding_file, embeddings)
    return embedding_file

def process_pdf(file_path):
    """Extract text, tables, images, split into chunks, store embeddings, and save in FAISS."""
    if os.path.exists(file_path):
        print(f"Processing {file_path}...")
        text_content = extract_text(file_path)
        table_content = extract_tables(file_path)
        extract_images(file_path)
        
        # Store chunks separately without merging text and tables
        text_chunks = split_text_recursive(text_content)
        table_chunks = split_text_recursive(table_content)
        
        # Save chunked text files
        text_chunk_file = os.path.join(chunk_folder, os.path.basename(file_path) + "_text_chunks.txt")
        with open(text_chunk_file, "w", encoding="utf-8") as file:
            for i, chunk in enumerate(text_chunks):
                file.write(f"Chunk {i+1}:\n{chunk}\n\n")
        
        table_chunk_file = os.path.join(chunk_folder, os.path.basename(file_path) + "_table_chunks.txt")
        with open(table_chunk_file, "w", encoding="utf-8") as file:
            for i, chunk in enumerate(table_chunks):
                file.write(f"Chunk {i+1}:\n{chunk}\n\n")
        
        # Generate embeddings only for text chunks
        chunk_embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
        
        # Store embeddings
        embedding_file = store_embeddings(os.path.basename(file_path), chunk_embeddings)
        print(f"Embeddings saved to {embedding_file}")
        
        # Store in FAISS
        global index
        index.add(chunk_embeddings)
        print(f"Stored {len(text_chunks)} text chunks in FAISS.")
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