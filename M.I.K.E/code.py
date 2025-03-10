import os
import subprocess
import shutil
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename

# Ensure KB folder exists
os.makedirs("KB", exist_ok=True)

template = """ 
Answer question below

Here is the conversation history: {context}

Question: {question}

Answer:

"""    

model = OllamaLLM(model='mistral')
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model 

def call_pdf_extract(file_name):
    script_path = "pdfextrct.py"
    file_path = os.path.join("KB", file_name)
    
    if os.path.exists(file_path):
        print(f"Calling PDF extraction for {file_name}...")
        subprocess.run(["python", script_path, file_path])
    else:
        print("File not found in KB folder.")

def clear_kb():
    for folder in ["KB", "extracted_images", "extracted_text", "extracted_tables", "split_chunks" , "data_embeddings"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"{folder} and all extracted files deleted.")
    os.makedirs("KB", exist_ok=True)

def handle_convo():
    context = ""
    print("Welcome to AI ChatBot, Type 'exit' to quit")
    
    while True:
        user_input = input("Ask away: ")
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "clear":
            context = ""
            clear_kb()
            shutil.rmtree("KB")  # Remove the KB directory and its contents
            os.makedirs("KB")  # Recreate the KB directory
            print("Memory and all extracted files cleared")
            continue
        
        if user_input.lower() == "upload":
            file_path = input("Enter file path: ")
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                unique_file_name = get_unique_filename("KB", file_name)
                dest_path = os.path.join("KB", unique_file_name)
                with open(file_path, "rb") as file:
                    content = file.read()
                with open(dest_path, "wb") as storage:
                    storage.write(content)
                print(f"File uploaded successfully to KB/{unique_file_name}!")
                call_pdf_extract(unique_file_name)  # Call PDF extraction function
            else:
                print("File not found. Please try again.")
            continue
        
        result = chain.invoke({"context": context, "question": user_input})
        print("AI: ", result)
        context += f"\nUser: {user_input}\nAI : {result}"
    
if __name__ == "__main__":
    handle_convo()
