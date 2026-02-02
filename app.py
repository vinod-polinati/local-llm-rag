"""
Main CLI chatbot application for the M.I.K.E RAG system.
Provides document upload, retrieval, and AI-powered Q&A functionality.
Optimized for low latency with lazy loading and caching.
"""

import os
import subprocess
import shutil
import sys

import numpy as np
import faiss
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

from config import settings
from logger import logger
from validators import validate_file, validate_filename, ValidationError
from retriever import HybridRetriever, BM25Index, create_hybrid_retriever


class ModelManager:
    """
    Singleton manager for lazy-loading and caching models.
    Significantly reduces startup time and memory usage.
    """

    _instance = None
    _embedding_model: SentenceTransformer | None = None
    _llm_chain = None
    _faiss_index: faiss.IndexFlatL2 | None = None
    _retriever: HybridRetriever | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self._embedding_model is None:
            logger.info("Loading embedding model (first use)...")
            self._embedding_model = SentenceTransformer(settings.embedding_model)
            logger.info(f"Loaded embedding model: {settings.embedding_model}")
        return self._embedding_model

    @property
    def faiss_index(self) -> faiss.IndexFlatL2:
        """Load or create FAISS index."""
        if self._faiss_index is None:
            try:
                if os.path.exists(settings.faiss_index_path):
                    self._faiss_index = faiss.read_index(settings.faiss_index_path)
                    logger.info(f"Loaded FAISS index with {self._faiss_index.ntotal} embeddings")
                else:
                    self._faiss_index = faiss.IndexFlatL2(settings.embedding_dimension)
                    logger.info("Created new FAISS index")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self._faiss_index = faiss.IndexFlatL2(settings.embedding_dimension)
        return self._faiss_index

    @faiss_index.setter
    def faiss_index(self, value: faiss.IndexFlatL2):
        """Update FAISS index."""
        self._faiss_index = value
        # Reset retriever to use new index
        self._retriever = None

    @property
    def llm_chain(self):
        """Lazy load LLM chain (only when first query is made)."""
        if self._llm_chain is None:
            logger.info("Initializing LLM (first query)...")
            template = """You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
- Answer based on the context provided
- If the context doesn't contain enough information, say so
- Be concise but thorough

Answer:"""
            try:
                model = OllamaLLM(model=settings.llm_model)
                prompt = ChatPromptTemplate.from_template(template)
                self._llm_chain = prompt | model
                logger.info(f"Initialized LLM: {settings.llm_model}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise
        return self._llm_chain

    @property
    def retriever(self) -> HybridRetriever:
        """Get or create hybrid retriever."""
        if self._retriever is None:
            self._retriever = create_hybrid_retriever(
                self.embedding_model,
                self.faiss_index
            )
        return self._retriever

    def reload_index(self):
        """Reload FAISS index and retriever after document updates."""
        if os.path.exists(settings.faiss_index_path):
            self._faiss_index = faiss.read_index(settings.faiss_index_path)
            self._retriever = None  # Will be recreated on next access
            logger.info(f"Reloaded FAISS index with {self._faiss_index.ntotal} embeddings")

    def clear(self):
        """Clear all cached models and indexes."""
        self._faiss_index = faiss.IndexFlatL2(settings.embedding_dimension)
        self._retriever = None
        logger.info("Cleared model manager cache")


# Global model manager
models = ModelManager()


def initialize_directories() -> None:
    """Create required directories if they don't exist."""
    for folder in [settings.kb_folder, settings.chunks_folder, settings.embeddings_folder]:
        os.makedirs(folder, exist_ok=True)
    logger.debug("Initialized required directories")


def get_unique_filename(directory: str, filename: str) -> str:
    """Ensure unique filenames to avoid overwriting."""
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename


def call_pdf_extract(file_name: str) -> bool:
    """
    Call the PDF extraction script and reload indexes after processing.

    Args:
        file_name: Name of the file in the KB folder.

    Returns:
        True if extraction succeeded, False otherwise.
    """
    script_path = "pdfextrct.py"
    file_path = os.path.join(settings.kb_folder, file_name)

    if not os.path.exists(file_path):
        logger.error(f"File not found in KB folder: {file_name}")
        return False

    try:
        logger.info(f"Starting PDF extraction for: {file_name}")
        result = subprocess.run(
            [sys.executable, script_path, file_path],
            timeout=settings.pdf_extraction_timeout,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"PDF extraction failed: {result.stderr}")
            return False

        logger.debug(f"Extraction output: {result.stdout}")

        # Reload indexes
        models.reload_index()
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"PDF extraction timed out after {settings.pdf_extraction_timeout}s")
        return False
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return False


def retrieve_context(query: str) -> str:
    """
    Retrieve the most relevant chunks using hybrid search.

    Args:
        query: The search query.

    Returns:
        Concatenated text from relevant chunks.
    """
    if models.faiss_index.ntotal == 0:
        logger.warning("FAISS index is empty - no documents uploaded")
        return "No documents in the knowledge base. Please upload a document first."

    try:
        # Use hybrid retriever
        results = models.retriever.retrieve(query, top_k=settings.retrieval_top_k)

        if not results:
            logger.info("No relevant chunks found for query")
            return "No relevant context found."

        # Get top chunks after re-ranking
        top_results = results[:settings.rerank_top_k]

        retrieved_chunks = []
        for idx, score in top_results:
            chunk_path = os.path.join(settings.chunks_folder, f"chunk_{idx}.txt")
            if os.path.exists(chunk_path):
                try:
                    with open(chunk_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        retrieved_chunks.append(f"[Relevance: {score:.2f}]\n{content}")
                except IOError as e:
                    logger.warning(f"Failed to read chunk {idx}: {e}")

        logger.debug(f"Retrieved {len(retrieved_chunks)} chunks for query")

        # Log cache stats periodically
        cache_stats = models.retriever.get_cache_stats()
        if cache_stats:
            logger.debug(f"Embedding cache: {cache_stats}")

        return "\n\n---\n\n".join(retrieved_chunks)

    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        return "Error retrieving context. Please try again."


def clear_kb() -> None:
    """Clear the knowledge base and all extracted files."""
    try:
        for folder in [settings.kb_folder, settings.chunks_folder, settings.embeddings_folder]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                logger.info(f"Deleted folder: {folder}")

        os.makedirs(settings.kb_folder, exist_ok=True)

        # Clear indexes
        models.clear()

        for path in [settings.faiss_index_path, settings.bm25_index_path]:
            if os.path.exists(path):
                os.remove(path)

        logger.info("Knowledge base cleared successfully")

    except Exception as e:
        logger.error(f"Failed to clear knowledge base: {e}")


def handle_upload() -> None:
    """Handle file upload from user."""
    file_path = input("📂 Enter file path: ").strip()

    if not file_path:
        print("❌ No file path provided.")
        return

    if not os.path.exists(file_path):
        print("❌ File not found. Please check the path and try again.")
        logger.warning(f"Upload failed - file not found: {file_path}")
        return

    try:
        # Validate the file
        filename = os.path.basename(file_path)
        validate_filename(filename)
        validate_file(file_path=file_path)

        # Copy to KB folder
        unique_file_name = get_unique_filename(settings.kb_folder, filename)
        dest_path = os.path.join(settings.kb_folder, unique_file_name)
        shutil.copy(file_path, dest_path)

        print(f"✅ File uploaded successfully: {unique_file_name}")
        logger.info(f"File uploaded: {unique_file_name}")

        # Process the PDF
        print("⏳ Processing document...")
        if call_pdf_extract(unique_file_name):
            print("✅ Document processed and indexed.")
        else:
            print("⚠️ Document processing failed. See logs for details.")

    except ValidationError as e:
        print(f"❌ Validation error: {e}")
        logger.warning(f"Upload validation failed: {e}")
    except IOError as e:
        print(f"❌ File error: {e}")
        logger.error(f"Upload IO error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.exception(f"Unexpected upload error: {e}")


def handle_query(user_input: str) -> None:
    """Process a user query and generate response."""
    try:
        print("🔍 Searching knowledge base...")
        context = retrieve_context(user_input)

        print("🧠 Generating response...")
        result = models.llm_chain.invoke({"context": context, "question": user_input})
        print("\n🧠 AI:", result)

        # Save FAISS index periodically
        faiss.write_index(models.faiss_index, settings.faiss_index_path)

    except Exception as e:
        print("❌ Error generating response. Please try again.")
        logger.exception(f"Query processing error: {e}")


def handle_convo() -> None:
    """Handle the chatbot's main conversation loop."""
    print("🤖 Welcome to M.I.K.E AI ChatBot!")
    print("   Commands: 'upload' | 'clear' | 'stats' | 'exit'")
    print("-" * 40)

    # Initialize directories but don't load heavy models yet
    initialize_directories()

    while True:
        try:
            user_input = input("\n💬 Ask away: ").strip()

            if not user_input:
                continue

            command = user_input.lower()

            if command == "exit":
                print("👋 Goodbye!")
                logger.info("User exited the chatbot")
                break

            if command == "clear":
                clear_kb()
                print("🧹 Knowledge base cleared.")
                continue

            if command == "upload":
                handle_upload()
                continue

            if command == "stats":
                cache_stats = models.retriever.get_cache_stats()
                print(f"📊 FAISS Index: {models.faiss_index.ntotal} embeddings")
                if cache_stats:
                    print(f"📊 Embedding Cache: {cache_stats}")
                continue

            # Regular query
            handle_query(user_input)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            logger.info("User interrupted the chatbot")
            break
        except EOFError:
            break
        except Exception as e:
            print("❌ An error occurred. Please try again.")
            logger.exception(f"Conversation loop error: {e}")


if __name__ == "__main__":
    try:
        handle_convo()
    except Exception as e:
        logger.critical(f"Application crashed: {e}")
        sys.exit(1)
