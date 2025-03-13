# local-llm-rag

A local Retrieval-Augmented Generation (RAG) system using FAISS, Sentence Transformers, and Ollama's Mistral model for chatbot-style interactions with uploaded documents.

## Features

- Local document storage and retrieval using FAISS
- PDF extraction for knowledge base ingestion
- Embedding generation with `sentence-transformers/all-MiniLM-L6-v2`
- Retrieval-based chatbot using Ollama's Mistral model
- Command-line interface for interaction

## Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- pip

### Clone the Repository
```bash
$ git clone https://github.com/vinod-polinati/local-llm-rag.git
$ cd local-llm-rag
$ cd M.I.K.E
```

### Activate Virtual Environment
```
$ python3 env env
```

### Install Dependencies
```bash
$ pip install -r requirements.txt
```

## Usage

### Running the Chatbot
```bash
$ python app.py
```

### Commands
- **Ask questions**: Simply type your question.
- **Upload documents**: Type `upload` and enter the file path.
- **Clear KB**: Type `clear` to remove all stored data.
- **Exit**: Type `exit` to quit.

## Structure
```
local-llm-rag/
│── KB/                  # Stored PDFs
│── split_chunks/        # Extracted text chunks
│── data_embeddings/     # Embedding storage
│── faiss_index.idx      # FAISS index file
│── main.py              # Main chatbot script
│── pdfextract.py        # PDF extraction script
│── requirements.txt     # Dependencies
```

## Acknowledgments
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Ollama](https://ollama.ai/)

## License
This project is licensed under the MIT License.

