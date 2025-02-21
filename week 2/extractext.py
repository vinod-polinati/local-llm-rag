import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

# Example usage:
pdf_text = extract_text_from_pdf("/Users/vinod/Desktop/mike/sample.pdf")
print(pdf_text[:500])  # Print first 500 characters