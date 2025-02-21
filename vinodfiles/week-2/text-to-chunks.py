import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

# Function to split text into chunks
def split_text(text, chunk_size=100, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

# Example usage
pdf_text = extract_text_from_pdf("/Users/vinod/Desktop/mike/sample.pdf")
chunks = split_text(pdf_text)

print(f"Total chunks: {len(chunks)}")
print(chunks[:2])  # Print first two chunks  