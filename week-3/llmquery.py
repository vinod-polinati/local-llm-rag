import ollama

def query_llm(prompt):
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# Example usage:
print(query_llm("Explain mitochondria in simple terms."))