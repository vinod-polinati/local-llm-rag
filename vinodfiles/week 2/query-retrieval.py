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

# Step 4: Store embeddings in FAISS index
dimension = chunk_embeddings.shape[1]  # 384 for MiniLM
index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance

# Convert embeddings to FAISS format
faiss_data = np.array(chunk_embeddings, dtype=np.float32)
index.add(faiss_data)

print("Embeddings stored in FAISS index!")

# Step 5: Search for similar chunks
def search_similar_chunks(query, top_k=3):
    query_embedding = model.encode([query])  # Convert query to embedding
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    
    return [chunks[i] for i in indices[0]]

# Example usage:
query = "What is mitochondria?"
relevant_chunks = search_similar_chunks(query)
print("\n".join(relevant_chunks))  # Print retrieved document sections
