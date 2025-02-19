import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer

# Define folders
chunk_folder = "split_chunks"
embedding_folder = "embeddings"

# Ensure embedding folder exists
os.makedirs(embedding_folder, exist_ok=True)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def process_chunk_file(file_name):
    """Processes a single chunk file and generates embeddings."""
    file_path = os.path.join(chunk_folder, file_name)
    
    # Read chunks from the file
    with open(file_path, "r", encoding="utf-8") as file:
        chunks = [line.strip() for line in file.readlines() if line.strip()]
    
    if not chunks:
        print(f"No valid text found in {file_name}")
        return

    # Generate embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True)

    # Save embeddings
    embedding_file = os.path.join(embedding_folder, file_name.replace("_chunks.txt", "_embeddings.npy"))
    np.save(embedding_file, embeddings)

    print(f"Saved embeddings for {file_name} in {embedding_file}")

def monitor_split_chunks():
    """Continuously watches the split_chunks folder for new files and processes them."""
    processed_files = set()

    while True:
        chunk_files = {f for f in os.listdir(chunk_folder) if f.endswith("_chunks.txt")}

        # Identify new files
        new_files = chunk_files - processed_files

        for file_name in new_files:
            process_chunk_file(file_name)
            processed_files.add(file_name)

        time.sleep(5)  # Check for new files every 5 seconds

if __name__ == "__main__":
    print("Monitoring split_chunks for new files...")
    monitor_split_chunks()
