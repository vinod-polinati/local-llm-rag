# M.I.K.E - Local AI Document Assistant

[![CI](https://github.com/vinod-polinati/local-llm-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/vinod-polinati/local-llm-rag/actions/workflows/ci.yml)

A **100% local** Retrieval-Augmented Generation (RAG) system. Chat with your documents privately—no data leaves your machine.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔒 **Privacy-First** | All processing happens locally (embeddings, LLM, storage) |
| ⚡ **Fast** | Lazy loading, embedding cache, hybrid search |
| 🎯 **Accurate** | BM25 + FAISS hybrid search with semantic chunking |
| 🐳 **Dockerized** | One command to run anywhere |
| ✅ **Tested** | 32 unit tests with CI/CD |

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
docker-compose up
# Open http://localhost:8501
```

### Option 2: Local Installation
```bash
# Clone
git clone https://github.com/vinod-polinati/local-llm-rag.git
cd local-llm-rag

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Ollama (separate terminal)
ollama serve
ollama pull mistral

# Run
streamlit run app-st.py    # Web UI
python3 app.py             # CLI
```

## 📖 Usage

### Web UI (Streamlit)
1. Upload PDF via sidebar
2. Ask questions in the text input
3. View retrieved context in expander

### CLI Commands
| Command | Action |
|---------|--------|
| `upload` | Add a PDF to knowledge base |
| `clear` | Reset knowledge base |
| `stats` | Show cache and index stats |
| `exit` | Quit |

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   PDF       │────▶│  Semantic    │────▶│   FAISS +   │
│   Upload    │     │  Chunking    │     │   BM25      │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐            │
│   Answer    │◀────│   Mistral    │◀───────────┘
│             │     │   (Ollama)   │     Hybrid Search
└─────────────┘     └──────────────┘
```

## ⚙️ Configuration

Copy `.env.example` to `.env` and customize:

```env
# LLM
LLM_MODEL=mistral              # or phi3, gemma:2b

# Performance
ENABLE_EMBEDDING_CACHE=true
USE_HYBRID_SEARCH=true
USE_SEMANTIC_CHUNKING=true

# Retrieval
RETRIEVAL_TOP_K=5
RERANK_TOP_K=3
```

## 🧪 Testing

```bash
pytest tests/ -v
# 32 passed in ~2s
```

## 📁 Project Structure

```
local-llm-rag/
├── app.py              # CLI interface
├── app-st.py           # Streamlit web UI
├── pdfextrct.py        # PDF processing + chunking
├── retriever.py        # Hybrid search (BM25 + FAISS)
├── config.py           # Centralized settings
├── validators.py       # File validation
├── logger.py           # Structured logging
├── tests/              # Unit tests
├── Dockerfile          # Container build
├── docker-compose.yml  # App + Ollama
└── .github/workflows/  # CI/CD
```

## 🛠️ Tech Stack

- **LLM**: Ollama (Mistral)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Keyword Search**: BM25
- **Web UI**: Streamlit
- **Config**: Pydantic Settings

## License

MIT License

## Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [LangChain](https://langchain.com/) - LLM orchestration
