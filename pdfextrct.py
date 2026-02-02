"""
PDF extraction module for the M.I.K.E RAG system.
Extracts text from PDFs (with OCR fallback), uses semantic chunking,
generates embeddings, and stores them in FAISS with BM25 index.
"""

import os
import re
import sys

import fitz  # PyMuPDF
import numpy as np
import faiss
from pdf2image import convert_from_path
import pytesseract
from sentence_transformers import SentenceTransformer

from config import settings
from logger import logger
from retriever import BM25Index


def initialize_directories() -> None:
    """Create required directories if they don't exist."""
    os.makedirs(settings.chunks_folder, exist_ok=True)
    os.makedirs(settings.embeddings_folder, exist_ok=True)


def load_embedding_model() -> SentenceTransformer:
    """Load the sentence transformer embedding model."""
    try:
        model = SentenceTransformer(settings.embedding_model)
        logger.debug(f"Loaded embedding model: {settings.embedding_model}")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def load_faiss_index() -> faiss.IndexFlatL2:
    """Load existing FAISS index or create a new one."""
    try:
        if os.path.exists(settings.faiss_index_path):
            index = faiss.read_index(settings.faiss_index_path)
            logger.debug(f"Loaded FAISS index with {index.ntotal} embeddings")
            return index
    except Exception as e:
        logger.warning(f"Failed to load FAISS index, creating new: {e}")

    return faiss.IndexFlatL2(settings.embedding_dimension)


def extract_text_with_ocr(pdf_path: str) -> str:
    """
    Extract text from PDF with OCR fallback for scanned pages.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text content.

    Raises:
        Exception: If PDF cannot be opened or processed.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF: {pdf_path} - {e}")
        raise

    full_text = ""

    for page_num, page in enumerate(doc):
        try:
            text = page.get_text()

            if not text.strip():
                # No text found - fall back to OCR
                logger.info(f"Running OCR on page {page_num + 1}")
                try:
                    images = convert_from_path(
                        pdf_path,
                        first_page=page_num + 1,
                        last_page=page_num + 1
                    )
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0])
                        full_text += ocr_text + "\n"
                        logger.debug(f"OCR extracted {len(ocr_text)} chars from page {page_num + 1}")
                except Exception as ocr_error:
                    logger.warning(f"OCR failed on page {page_num + 1}: {ocr_error}")
            else:
                full_text += text + "\n"
                logger.debug(f"Extracted {len(text)} chars from page {page_num + 1}")

        except Exception as e:
            logger.warning(f"Failed to extract page {page_num + 1}: {e}")
            continue

    doc.close()

    if not full_text.strip():
        logger.warning(f"No text extracted from PDF: {pdf_path}")

    return full_text


def semantic_chunk_text(text: str, target_size: int | None = None, overlap: int | None = None) -> list[str]:
    """
    Split text into semantically meaningful chunks based on sentences.
    This produces better retrieval results than fixed-size chunking.

    Args:
        text: The text to chunk.
        target_size: Target size for each chunk in characters.
        overlap: Overlap between chunks in characters.

    Returns:
        List of text chunks.
    """
    if target_size is None:
        target_size = settings.chunk_size
    if overlap is None:
        overlap = settings.chunk_overlap

    if not text or not text.strip():
        return []

    # Split into sentences using regex
    # Handles: periods, question marks, exclamation marks
    # Preserves abbreviations like "Dr.", "Mr.", "e.g.", etc.
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text)

    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)

        # If adding this sentence would exceed target, save current chunk
        if current_size + sentence_size > target_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

            # Keep some sentences for overlap
            overlap_size = 0
            overlap_sentences = []
            for s in reversed(current_chunk):
                if overlap_size + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_size += len(s)
                else:
                    break

            current_chunk = overlap_sentences
            current_size = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_size += sentence_size

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text)

    logger.debug(f"Semantic chunking: {len(sentences)} sentences -> {len(chunks)} chunks")
    return chunks


def fixed_chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """
    Split text into fixed-size overlapping chunks (legacy method).

    Args:
        text: The text to chunk.
        chunk_size: Size of each chunk in characters.
        overlap: Overlap between chunks in characters.

    Returns:
        List of text chunks.
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if overlap is None:
        overlap = settings.chunk_overlap

    if not text or not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    logger.debug(f"Fixed chunking: {len(text)} chars -> {len(chunks)} chunks")
    return chunks


def chunk_text(text: str) -> list[str]:
    """
    Chunk text using the configured method.

    Args:
        text: The text to chunk.

    Returns:
        List of text chunks.
    """
    if settings.use_semantic_chunking:
        return semantic_chunk_text(text)
    else:
        return fixed_chunk_text(text)


def process_pdf(pdf_path: str) -> bool:
    """
    Process a PDF: extract text, chunk, embed, and store in FAISS + BM25.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        True if processing succeeded, False otherwise.
    """
    logger.info(f"Starting PDF processing: {pdf_path}")

    # Validate file exists
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return False

    try:
        # Initialize
        initialize_directories()
        embedding_model = load_embedding_model()
        faiss_index = load_faiss_index()

        # Load or create BM25 index
        bm25_index = BM25Index()
        bm25_index.load(settings.bm25_index_path)

        # Extract text
        logger.info("Extracting text from PDF...")
        text = extract_text_with_ocr(pdf_path)

        if not text.strip():
            logger.warning("No text could be extracted from PDF")
            return False

        # Chunk text (semantic or fixed based on config)
        logger.info("Chunking text...")
        chunks = chunk_text(text)

        if not chunks:
            logger.warning("No chunks created from extracted text")
            return False

        # Process each chunk
        logger.info(f"Processing {len(chunks)} chunks...")
        chunks_processed = 0
        new_chunks = []

        for i, chunk in enumerate(chunks):
            try:
                # Save chunk to file
                chunk_idx = faiss_index.ntotal + i
                chunk_path = os.path.join(settings.chunks_folder, f"chunk_{chunk_idx}.txt")

                with open(chunk_path, "w", encoding="utf-8") as f:
                    f.write(chunk)

                # Generate and store embedding
                embedding = embedding_model.encode([chunk])[0]
                embedding = np.array(embedding, dtype="float32").reshape(1, -1)
                faiss_index.add(embedding)

                new_chunks.append(chunk)
                chunks_processed += 1

            except Exception as e:
                logger.warning(f"Failed to process chunk {i}: {e}")
                continue

        # Update BM25 index with new chunks
        if new_chunks:
            bm25_index.add_documents(new_chunks)
            bm25_index.save(settings.bm25_index_path)
            logger.info(f"Updated BM25 index with {len(new_chunks)} new chunks")

        # Save updated FAISS index
        try:
            faiss.write_index(faiss_index, settings.faiss_index_path)
            logger.info(f"Saved FAISS index with {faiss_index.ntotal} total embeddings")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            return False

        logger.info(f"Successfully processed {chunks_processed}/{len(chunks)} chunks from PDF")
        return chunks_processed > 0

    except Exception as e:
        logger.exception(f"PDF processing failed: {e}")
        return False


def main() -> int:
    """Main entry point for CLI usage."""
    if len(sys.argv) != 2:
        print("Usage: python3 pdfextrct.py <PDF_PATH>")
        return 1

    pdf_path = sys.argv[1]

    try:
        success = process_pdf(pdf_path)
        return 0 if success else 1
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
