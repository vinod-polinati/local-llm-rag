import os
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import pdfplumber

# Ensure KB folder exists
kb_folder = "KB"
os.makedirs(kb_folder, exist_ok=True)

def extract_text(file_path, output_folder="extracted_text"):
    try:
        os.makedirs(output_folder, exist_ok=True)
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        text_filename = os.path.join(output_folder, os.path.basename(file_path) + ".txt")
        with open(text_filename, "w", encoding="utf-8") as text_file:
            text_file.write(text)
        
        return f"Extracted text saved to '{text_filename}'"
    except Exception as e:
        return f"Error extracting text: {e}"

def extract_tables(file_path, output_folder="extracted_tables"):
    try:
        os.makedirs(output_folder, exist_ok=True)
        tables_data = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    tables_data.extend(tables)
        
        table_filename = os.path.join(output_folder, os.path.basename(file_path) + ".csv")
        with open(table_filename, "w", encoding="utf-8") as table_file:
            for table in tables_data:
                for row in table:
                    # Convert None values to empty strings
                    row = [str(cell) if cell is not None else "" for cell in row]
                    table_file.write(",".join(row) + "\n")
        
        return f"Extracted tables saved to '{table_filename}'"
    except Exception as e:
        return f"Error extracting tables: {e}"

def extract_images(file_path, output_folder="extracted_images"):
    try:
        os.makedirs(output_folder, exist_ok=True)
        doc = fitz.open(file_path)
        image_count = 0
        
        for page_num, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img = Image.open(BytesIO(image_bytes))
                img_filename = f"{output_folder}/page_{page_num + 1}_img_{img_index + 1}.png"
                img.save(img_filename, "PNG")
                image_count += 1
        
        return f"Extracted {image_count} images to '{output_folder}'"
    except Exception as e:
        return f"Error extracting images: {e}"
 
def process_pdf(file_path):
    if os.path.exists(file_path):
        print(f"Processing {file_path}...")
        text_result = extract_text(file_path)
        print(text_result)
        
        table_result = extract_tables(file_path)
        print(table_result)
        
        image_result = extract_images(file_path)
        print(image_result)
    else:
        print(f"File not found: {file_path}")

def process_all_pdfs():
    files = [f for f in os.listdir(kb_folder) if f.lower().endswith(".pdf")]
    if not files:
        print("No PDF files found in KB folder.")
        return
    
    for file_name in files:
        file_path = os.path.join(kb_folder, file_name)
        process_pdf(file_path)

# Automatically process all PDFs in KB folder
if __name__ == "__main__":
    process_all_pdfs()