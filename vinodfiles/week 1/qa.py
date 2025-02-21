import ollama

def ask_llm(query):
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": query}])
    return response['message']['content']  # Extracts only the assistant's response

print(ask_llm("What is Deep Learning?"))
