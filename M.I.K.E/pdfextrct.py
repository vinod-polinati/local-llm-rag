import os
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import pdfplumber
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Define folders
kb_folder = "KB"
text_folder = "extracted_text"
table_folder = "extracted_tables"
image_folder = "extracted_images"
chunk_folder = "split_chunks"
embedding_folder = "embeddings"

# Ensure required folders exist
os.makedirs(kb_folder, exist_ok=True)
os.makedirs(text_folder, exist_ok=True)
os.makedirs(table_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)
os.makedirs(chunk_folder, exist_ok=True)
os.makedirs(embedding_folder, exist_ok=True)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(file_path):
    """Extract text from a PDF."""
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
    """Extract tables from a PDF and save them as CSV."""
    try:
        tables_data = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    tables_data.extend(tables)
        
        table_filename = os.path.join(table_folder, os.path.basename(file_path) + ".csv")
        with open(table_filename, "w", encoding="utf-8") as table_file:
            for table in tables_data:
                for row in table:
                    row = [str(cell) if cell is not None else "" for cell in row]
                    table_file.write(",".join(row) + "\n")
        
        return "\n".join([",".join(row) for table in tables_data for row in table])
    except Exception as e:
        print(f"Error extracting tables: {e}")
        return ""

def extract_images(file_path, image_folder="extracted_images"):
    """Extract images from a PDF using PyMuPDF and save them."""
    try:
        os.makedirs(image_folder, exist_ok=True)
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img = Image.open(BytesIO(image_bytes))
                img.save(os.path.join(image_folder, f"{os.path.basename(file_path)}_page{page_num+1}_img{img_index+1}.png"), "PNG")
                print(f"Saved image: {os.path.basename(file_path)}_page{page_num+1}_img{img_index+1}.png")
    except Exception as e:
        print(f"Error extracting images from {file_path}: {e}")

def split_text_recursive(text, chunk_size=500, chunk_overlap=50):
    """Split text using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def encode_chunks(file_name, chunks):
    """Convert chunks to embeddings and save them."""
    if not chunks:
        print(f"No valid chunks found in {file_name}")
        return
    
    # Generate embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True)

    # Save embeddings
    embedding_file = os.path.join(embedding_folder, file_name.replace(".pdf", "_embeddings.npy"))
    np.save(embedding_file, embeddings)

    print(f"Saved embeddings for {file_name} in {embedding_file}")

def process_pdf(file_path):
    """Extract text, tables, images, split into chunks, and generate embeddings."""
    if os.path.exists(file_path):
        print(f"Processing {file_path}...")

        text_content = extract_text(file_path)
        table_content = extract_tables(file_path)
        extract_images(file_path)

        # Merge text and tables for chunking
        full_content = text_content + "\n" + table_content
        if full_content.strip():  # Only split if there's content
            chunks = split_text_recursive(full_content)

            # Save chunks
            chunk_output_file = os.path.join(chunk_folder, os.path.basename(file_path) + "_chunks.txt")
            with open(chunk_output_file, "w", encoding="utf-8") as file:
                for i, chunk in enumerate(chunks):
                    file.write(f"Chunk {i+1}:\n{chunk}\n\n")

            print(f"Split chunks saved to '{chunk_output_file}'")

            # Encode chunks immediately after splitting
            encode_chunks(os.path.basename(file_path), chunks)
        else:
            print("No text or table data found to split.")

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

# Automatically process PDFs when script runs
if __name__ == "__main__":
    process_all_pdfs()