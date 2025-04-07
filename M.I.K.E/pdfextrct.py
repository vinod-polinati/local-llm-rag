import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Configs
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dimension = 384
faiss_index_path = "faiss_index.idx"

# Ensure output folders exist
os.makedirs("split_chunks", exist_ok=True)
os.makedirs("data_embeddings", exist_ok=True)

# Load or create FAISS index
if os.path.exists(faiss_index_path):
    index = faiss.read_index(faiss_index_path)
else:
    index = faiss.IndexFlatL2(dimension)

def extract_text_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if not text.strip():
            # No text: fall back to OCR
            print(f"🔍 Running OCR on page {page_num + 1}")
            images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
            if images:
                ocr_text = pytesseract.image_to_string(images[0])
                full_text += ocr_text + "\n"
        else:
            full_text += text + "\n"
    return full_text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

def process_pdf(pdf_path):
    print(f"📄 Extracting from: {pdf_path}")
    text = extract_text_with_ocr(pdf_path)
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        chunk_path = f"split_chunks/chunk_{index.ntotal + i}.txt"
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(chunk)

        embedding = embedding_model.encode([chunk])[0]
        embedding = np.array(embedding, dtype="float32").reshape(1, -1)
        index.add(embedding)

    # Save updated index
    faiss.write_index(index, faiss_index_path)
    print(f"✅ Processed {len(chunks)} chunks and updated FAISS index.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 pdfextrct.py <PDF_PATH>")
    else:
        process_pdf(sys.argv[1])
