import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

pdf_text = extract_text_from_pdf("/Users/vinod/Desktop/mike/sample.pdf")

# Step 2: Split text into chunks
def split_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

chunks = split_text(pdf_text)
print(f"Total chunks: {len(chunks)}")

# Step 3: Convert text chunks into embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunks)

# Print first 5 embeddings (optional)
for i, embedding in enumerate(chunk_embeddings[:5]):  
    print(f"Embedding {i+1}: {embedding}\n")

print(f"Generated {len(chunk_embeddings)} embeddings!")

# Step 4: Store embeddings in FAISS index
dimension = chunk_embeddings.shape[1]  # 384 for MiniLM
index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance

# Convert embeddings to FAISS format
faiss_data = np.array(chunk_embeddings, dtype=np.float32)
index.add(faiss_data)

print("Embeddings stored in FAISS index!")