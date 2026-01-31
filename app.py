"""
Main CLI chatbot application for the M.I.K.E RAG system.
Provides document upload, retrieval, and AI-powered Q&A functionality.
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


def initialize_directories() -> None:
    """Create required directories if they don't exist."""
    for folder in [settings.kb_folder, settings.chunks_folder, settings.embeddings_folder]:
        os.makedirs(folder, exist_ok=True)
    logger.debug("Initialized required directories")


def load_faiss_index() -> faiss.IndexFlatL2:
    """Load existing FAISS index or create a new one."""
    try:
        if os.path.exists(settings.faiss_index_path):
            index = faiss.read_index(settings.faiss_index_path)
            logger.info(f"Loaded FAISS index with {index.ntotal} embeddings")
            return index
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")

    index = faiss.IndexFlatL2(settings.embedding_dimension)
    logger.info("Created new FAISS index")
    return index


def load_embedding_model() -> SentenceTransformer:
    """Load the sentence transformer embedding model."""
    try:
        model = SentenceTransformer(settings.embedding_model)
        logger.info(f"Loaded embedding model: {settings.embedding_model}")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def load_llm_chain():
    """Initialize the LLM and prompt chain."""
    template = """
Use the information below to answer the question.

Context: {context}

Question: {question}

Answer:
"""
    try:
        model = OllamaLLM(model=settings.llm_model)
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        logger.info(f"Initialized LLM chain with model: {settings.llm_model}")
        return chain
    except Exception as e:
        logger.error(f"Failed to initialize LLM chain: {e}")
        raise


# Initialize components
initialize_directories()
embedding_model = load_embedding_model()
index = load_faiss_index()
chain = load_llm_chain()


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
    Call the PDF extraction script and reload FAISS index after processing.

    Args:
        file_name: Name of the file in the KB folder.

    Returns:
        True if extraction succeeded, False otherwise.
    """
    global index
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

        # Reload FAISS index
        if os.path.exists(settings.faiss_index_path):
            index = faiss.read_index(settings.faiss_index_path)
            logger.info(f"Reloaded FAISS index with {index.ntotal} embeddings")
        else:
            logger.warning("FAISS index not found after extraction")
            return False

        return True

    except subprocess.TimeoutExpired:
        logger.error(f"PDF extraction timed out after {settings.pdf_extraction_timeout}s")
        return False
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return False


def retrieve_context(query: str, top_k: int | None = None) -> str:
    """
    Retrieve the most relevant chunks from FAISS.

    Args:
        query: The search query.
        top_k: Number of results to retrieve.

    Returns:
        Concatenated text from relevant chunks.
    """
    if top_k is None:
        top_k = settings.retrieval_top_k

    if index.ntotal == 0:
        logger.warning("FAISS index is empty - no documents uploaded")
        return "No documents in the knowledge base. Please upload a document first."

    try:
        query_embedding = embedding_model.encode([query])
        distances, indices = index.search(query_embedding, top_k)

        if indices[0][0] == -1:
            logger.info("No relevant chunks found for query")
            return "No relevant context found."

        retrieved_chunks = []
        for idx in indices[0]:
            if idx != -1:
                chunk_path = os.path.join(settings.chunks_folder, f"chunk_{idx}.txt")
                if os.path.exists(chunk_path):
                    try:
                        with open(chunk_path, "r", encoding="utf-8") as file:
                            retrieved_chunks.append(file.read())
                    except IOError as e:
                        logger.warning(f"Failed to read chunk {idx}: {e}")

        logger.debug(f"Retrieved {len(retrieved_chunks)} chunks for query")
        return "\n\n".join(retrieved_chunks)

    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        return "Error retrieving context. Please try again."


def clear_kb() -> None:
    """Clear the knowledge base and all extracted files."""
    global index

    try:
        for folder in [settings.kb_folder, settings.chunks_folder, settings.embeddings_folder]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                logger.info(f"Deleted folder: {folder}")

        os.makedirs(settings.kb_folder, exist_ok=True)

        index = faiss.IndexFlatL2(settings.embedding_dimension)

        if os.path.exists(settings.faiss_index_path):
            os.remove(settings.faiss_index_path)

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
        context = retrieve_context(user_input)
        result = chain.invoke({"context": context, "question": user_input})
        print("🧠 AI:", result)

        # Save FAISS index periodically
        faiss.write_index(index, settings.faiss_index_path)

    except Exception as e:
        print("❌ Error generating response. Please try again.")
        logger.exception(f"Query processing error: {e}")


def handle_convo() -> None:
    """Handle the chatbot's main conversation loop."""
    print("🤖 Welcome to M.I.K.E AI ChatBot!")
    print("   Commands: 'upload' | 'clear' | 'exit'")
    print("-" * 40)

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
