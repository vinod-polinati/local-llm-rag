import ollama

def ask_llm():
    query = input("Enter your question: ")  # Take user input
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": query}])
    print("\nLLM Response:\n", response['message']['content'])

# Ask at runtime in a loop
while True:
    ask_llm()