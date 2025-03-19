# **Machine Information Knowledge Extractor**

## **1. INTRODUCTION**
In the era of artificial intelligence, the need for intelligent and optimized chatbot systems is more critical than ever. This report explores an AI chatbot that leverages FAISS (Facebook AI Similarity Search) and Ollama to provide optimized retrieval and response generation. The chatbot is designed for efficient knowledge retrieval using an advanced vector search engine and a streamlined language model, ensuring fast and accurate responses.

The primary components of this system include:
- **FAISS HNSW** for fast approximate nearest-neighbor search.
- **Ollama LLM (Mistral 7B)** for generating intelligent responses.
- **Sentence Transformers** for embedding text into vector space.
- **Optimized retrieval** to ensure relevant information is fetched efficiently.

---

## **2. TECHNOLOGY OVERVIEW**

### **2.1 Vector Embeddings and Semantic Search**
Vector embeddings are numerical representations of text in a high-dimensional space, enabling efficient similarity matching. Sentence Transformers convert textual data into dense vectors, capturing semantic meanings beyond simple keyword matching. 

For example, the sentence **"AI is transforming industries"** would be mapped close to **"Artificial intelligence is changing businesses"** in vector space, even though the words differ.

FAISS (Facebook AI Similarity Search) is used to store and retrieve these embeddings efficiently. When a user queries the chatbot, their question is converted into an embedding and compared to stored embeddings to find the most relevant context.

#### **Retrieval-Augmented Generation (RAG) Workflow**
To enhance the chatbot's accuracy and relevance, a **Retrieval-Augmented Generation (RAG)** approach is implemented. The process follows these steps:
1. **Pre-processing:** User documents are collected, chunked into smaller parts, and embedded using an embedding model.
2. **Storage:** The generated embeddings are stored in a **Vector Database (FAISS)** for efficient retrieval.
3. **Retrieval:** When a user submits a query, the system converts it into an embedding and retrieves the most relevant context from the vector database.
4. **Augmentation:** The retrieved context is combined with the user's query to form a more informative input for the LLM.
5. **Response Generation:** The LLM processes the augmented query and provides a relevant response to the user.

## A visual representation of the **RAG workflow**:

![RAG Workflow](rag.png)

---

### **2.2 FAISS HNSW Indexing**
FAISS (Facebook AI Similarity Search) is a high-speed library for nearest-neighbor search, optimized for large-scale datasets. The system employs an HNSW (Hierarchical Navigable Small World) graph structure to accelerate search efficiency. By using **32 neighbors per node**, the retrieval process is significantly faster than brute-force approaches.

HNSW builds a graph where each node represents an embedding, and connections between nodes help navigate to the most relevant data points with fewer comparisons. This enables sublinear search time compared to traditional brute-force methods.

---

### **2.3 Ollama LLM (Mistral 7B)**
Ollama provides a lightweight large language model (LLM) suitable for AI-driven conversations. The **Mistral 7B model** was chosen for its balance between performance and efficiency. It delivers high-quality responses with reduced computational overhead compared to larger models like Llama 13B or GPT-4.

The model is optimized for:
- **Low latency inference**: Faster response generation.
- **Context-aware replies**: Generates intelligent responses based on retrieved embeddings.
- **Efficient token usage**: Processes user queries efficiently with minimal redundant computation.

---

### **2.4 Sentence Embeddings**
A pre-trained **SentenceTransformer** model ("all-MiniLM-L6-v2") is used to convert text into high-dimensional vectors. This enables fast and accurate similarity matching between queries and stored knowledge chunks.

Each text input is transformed into a **384-dimensional vector**, which is stored in FAISS for fast retrieval. When a new query arrives, it is converted into an embedding and compared against stored embeddings using FAISS to find the most relevant context.

---

## **3. IMPLEMENTATION**

### **3.1 Code Implementation**
Below is the Python implementation of the chatbot system:

```python
import os
import subprocess
import shutil
import torch
import numpy as np
import faiss
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

# ✅ Reduce embedding sequence length for speed
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_model.max_seq_length = 128  # Limits input size (faster execution)

dimension = 384
faiss_index_path = "faiss_index.idx"

# ✅ Use FAISS HNSW for fast approximate search
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors in graph
faiss.omp_set_num_threads(4)  # ✅ Limit FAISS CPU threads

# ✅ Use Mistral 7B for better response generation
model = OllamaLLM(model="mistral:7b")  # 🔥 High-performance open-source LLM
template = """ 
Use the information below to answer the question.

Context: {context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model 

# ✅ Optimized retrieval
def retrieve_context(query, top_k=2):
    if index.ntotal == 0:
        return "⚠️ No data available in the KB."
    
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    retrieved_chunks = []
    for idx in indices[0]:
        if idx != -1:
            chunk_path = f"split_chunks/chunk_{idx}.txt"
            if os.path.exists(chunk_path):
                with open(chunk_path, "r", encoding="utf-8") as file:
                    retrieved_chunks.append(file.read())

    return "\n\n".join(retrieved_chunks) if retrieved_chunks else "⚠️ No relevant context found."

def handle_convo():
    """Handles the chatbot's conversation loop."""
    print("🤖 Welcome to AI ChatBot, Type 'exit' to quit")

    while True:
        user_input = input("Ask away: ")

        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        # Retrieve relevant context from FAISS
        context = retrieve_context(user_input)

        # Generate response
        result = chain.invoke({"context": context, "question": user_input})
        print("🧠 AI: ", result)

        # Save FAISS index
        faiss.write_index(index, faiss_index_path)

if __name__ == "__main__":
    handle_convo()
```

---

## **4. FUNCTIONALITY AND FEATURES**
### **4.1 Key Features**
- **Optimized Search:** Uses FAISS HNSW for fast retrieval.
- **Efficient LLM Processing:** Uses Mistral 7B to enhance response quality.
- **Intelligent Context Retrieval:** Extracts relevant information before generating responses.
- **Low Latency:** Designed for real-time user interaction.

### **4.2 Workflow**
1. The user inputs a question.
2. FAISS searches for relevant context from stored knowledge.
3. The retrieved data is fed into the Ollama LLM (Mistral 7B) to generate a response.
4. The AI chatbot returns a meaningful answer to the user.
5. The FAISS index is updated and saved for future queries.

---

## **5. CONCLUSION**
This AI chatbot implementation efficiently combines **FAISS, Sentence Transformers, and Mistral 7B** to deliver **fast and relevant responses**. With **optimized search indexing** and **smaller yet powerful LLM models**, this system balances **performance and accuracy**, making it suitable for AI-driven conversational assistants in real-world applications.

---

## **REFERENCES**
1. FAISS Documentation: https://faiss.ai
2. LangChain Framework: https://python.langchain.com
3. Ollama LLM: https://ollama.ai
4. Sentence Transformers: https://www.sbert.net

