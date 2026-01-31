"""
Streamlit web interface for the M.I.K.E RAG system.
Provides document upload, retrieval, and AI-powered Q&A functionality.
"""

import os
import shutil

import faiss
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

from config import settings
from logger import logger
from validators import validate_file, ValidationError


def initialize_directories() -> None:
    """Create required directories if they don't exist."""
    for folder in [settings.kb_folder, settings.chunks_folder, settings.embeddings_folder]:
        os.makedirs(folder, exist_ok=True)


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    """Load the sentence transformer embedding model (cached)."""
    try:
        model = SentenceTransformer(settings.embedding_model)
        logger.info(f"Loaded embedding model: {settings.embedding_model}")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        st.error(f"Failed to load embedding model: {e}")
        raise


@st.cache_resource
def load_llm_chain():
    """Initialize the LLM and prompt chain (cached)."""
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
        st.error(f"Failed to connect to Ollama. Is it running?")
        raise


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
    Call the PDF extraction script and reload FAISS index.

    Args:
        file_name: Name of the file in the KB folder.

    Returns:
        True if extraction succeeded, False otherwise.
    """
    import subprocess
    import sys

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
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"PDF extraction timed out after {settings.pdf_extraction_timeout}s")
        return False
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return False


def retrieve_context(query: str, index: faiss.IndexFlatL2, embedding_model: SentenceTransformer) -> str:
    """Retrieve the most relevant chunks from FAISS."""
    if index.ntotal == 0:
        return "FAISS index is empty. Upload a document first."

    try:
        query_embedding = embedding_model.encode([query])
        distances, indices = index.search(query_embedding, settings.retrieval_top_k)

        if indices[0][0] == -1:
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
    try:
        for folder in [settings.kb_folder, settings.chunks_folder, settings.embeddings_folder]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                logger.info(f"Deleted folder: {folder}")

        os.makedirs(settings.kb_folder, exist_ok=True)

        if os.path.exists(settings.faiss_index_path):
            os.remove(settings.faiss_index_path)

        logger.info("Knowledge base cleared successfully")

    except Exception as e:
        logger.error(f"Failed to clear knowledge base: {e}")
        raise


# Initialize
initialize_directories()

# Streamlit UI Configuration
st.set_page_config(page_title="📚 M.I.K.E AI Chatbot", layout="centered")
st.title("- M.I.K.E -")
st.caption("📖 AI-Powered Document Assistant")

# Load models
try:
    embedding_model = load_embedding_model()
    chain = load_llm_chain()
except Exception:
    st.stop()

# Load FAISS index (not cached - needs to be reloaded after uploads)
index = load_faiss_index()

# Sidebar
with st.sidebar:
    st.header("📂 Document Management")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help=f"Maximum file size: {settings.max_file_size_mb}MB"
    )

    if uploaded_file:
        try:
            # Get file bytes for validation
            file_bytes = uploaded_file.getvalue()

            # Validate file
            validate_file(
                file_bytes=file_bytes,
                filename=uploaded_file.name,
                file_obj=uploaded_file
            )

            # Save file
            file_name = get_unique_filename(settings.kb_folder, uploaded_file.name)
            file_path = os.path.join(settings.kb_folder, file_name)

            with open(file_path, "wb") as f:
                f.write(file_bytes)

            logger.info(f"File uploaded: {file_name}")

            # Process PDF
            with st.spinner("Processing document..."):
                if call_pdf_extract(file_name):
                    st.success(f"✅ Uploaded and indexed: {file_name}")
                    # Reload index
                    index = load_faiss_index()
                else:
                    st.error("❌ Failed to process document. Check logs.")

        except ValidationError as e:
            st.error(f"❌ {e}")
            logger.warning(f"Upload validation failed: {e}")
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")
            logger.exception(f"Upload error: {e}")

    st.divider()

    # Clear button
    if st.button("🧹 Clear All Data", type="secondary"):
        try:
            clear_kb()
            st.success("✅ Knowledge base cleared")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to clear: {e}")

    st.divider()

    # List uploaded files
    st.subheader("📄 Uploaded Files")
    try:
        files = os.listdir(settings.kb_folder)
        if files:
            for file in files:
                st.markdown(f"- {file}")
        else:
            st.caption("No files uploaded yet")
    except Exception as e:
        st.error(f"Failed to list files: {e}")

st.divider()

# Main chat interface
user_question = st.text_input("💬 Ask a question about your documents:", key="user_input")

if user_question:
    try:
        with st.spinner("Thinking..."):
            context = retrieve_context(user_question, index, embedding_model)
            result = chain.invoke({"context": context, "question": user_question})

            logger.info(f"Query processed: {user_question[:50]}...")

        st.success("**Answer:**")
        st.write(result)

        # Save FAISS index
        try:
            faiss.write_index(index, settings.faiss_index_path)
        except Exception as e:
            logger.warning(f"Failed to save FAISS index: {e}")

    except Exception as e:
        st.error(f"❌ Error generating response: {e}")
        logger.exception(f"Query error: {e}")
