import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfextrct import process_pdf  # Ensure this module is properly imported and available

# Define folders
extracted_text_folder = "extracted_text"
extracted_tables_folder = "extracted_tables"
kb2_folder = "K2"

# Ensure necessary folders exist
os.makedirs(kb2_folder, exist_ok=True)

def initialize_folders():
    os.makedirs(extracted_text_folder, exist_ok=True)
    os.makedirs(extracted_tables_folder, exist_ok=True)

def wait_for_extraction():
    """Wait for extraction to complete before chunking."""
    print("Waiting for extracted files...")
    retries = 5
    while retries > 0:
        text_files = [f for f in os.listdir(extracted_text_folder) if f.endswith(".txt")]
        table_files = os.listdir(extracted_tables_folder)
        if text_files or table_files:
            print("Extraction complete. Proceeding to chunking.")
            return
        time.sleep(2)
        retries -= 1
    print("Warning: No extracted files detected! Skipping chunking.")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Adjust chunk size as needed
    chunk_overlap=50  # Adjust overlap as needed
)

def chunk_and_store():
    """Chunk extracted text and store in KB2."""
    text_files = [f for f in os.listdir(extracted_text_folder) if f.endswith(".txt")]
    if not text_files:
        print("No extracted text files found! Aborting chunking.")
        return
    
    # Process text files
    for file_name in text_files:
        file_path = os.path.join(extracted_text_folder, file_name)
        print(f"Processing file: {file_name}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                print(f"Warning: {file_name} is empty. Skipping.")
                continue
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue
        
        # Debugging: Print first 500 characters of extracted text
        print(f"First 500 chars of {file_name}: {text[:500]}")
        
        # Ensure text is valid before chunking
        if len(text) < 10:
            print(f"Warning: Extracted text in {file_name} is too short to split. Skipping.")
            continue
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        print(f"{len(chunks)} chunks created for {file_name}")
        
        if not chunks:
            print(f"Error: No chunks were created for {file_name}! Check text extraction.")
            continue
        
        # Save chunks to KB2
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(kb2_folder, f"{file_name}_chunk_{i}.txt")
            try:
                with open(chunk_file, "w", encoding="utf-8") as f:
                    f.write(chunk)
                print(f"Saved chunk {i} of {file_name} to {chunk_file}")
            except Exception as e:
                print(f"Error writing chunk {i} of {file_name}: {e}")
                continue
    
    print("Text chunking complete.")

# Main execution
if __name__ == "__main__":
    initialize_folders()
    wait_for_extraction()
    chunk_and_store()
    print("Chunking process completed.")