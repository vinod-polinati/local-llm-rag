import ollama

response = ollama.chat(
    model="mistral", 
    messages=[{"role": "user", "content": "What is Machine Learning?"}]
    )
print(response['message']['content'])  