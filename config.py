"""
Centralized configuration management using pydantic-settings.
All settings can be overridden via environment variables or .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings
    llm_model: str = "mistral"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Performance Settings
    enable_embedding_cache: bool = True
    embedding_cache_size: int = 1000
    lazy_load_llm: bool = True  # Don't load LLM until first query

    # RAG Settings
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 5  # Increased for re-ranking
    rerank_top_k: int = 3  # Final results after re-ranking
    use_hybrid_search: bool = True  # Combine BM25 + FAISS
    hybrid_alpha: float = 0.5  # Balance: 0=BM25 only, 1=FAISS only

    # File Settings
    max_file_size_mb: int = 50
    allowed_file_types: list[str] = ["application/pdf"]

    # Paths
    kb_folder: str = "KB"
    chunks_folder: str = "split_chunks"
    embeddings_folder: str = "data_embeddings"
    faiss_index_path: str = "faiss_index.idx"
    bm25_index_path: str = "bm25_index.pkl"
    log_folder: str = "logs"

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True

    # Processing
    pdf_extraction_timeout: int = 300  # seconds
    use_semantic_chunking: bool = True  # Sentence-aware chunking

    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB to bytes."""
        return self.max_file_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience singleton
settings = get_settings()
