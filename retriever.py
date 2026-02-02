"""
Advanced retrieval module with hybrid search and re-ranking.
Combines BM25 (keyword) and FAISS (semantic) search for better accuracy.
"""

import os
import pickle
import re
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import settings
from logger import logger


class EmbeddingCache:
    """LRU cache for query embeddings to reduce latency on repeated queries."""

    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> np.ndarray | None:
        """Get cached embedding for a query."""
        if query in self.cache:
            self.cache.move_to_end(query)
            self.hits += 1
            return self.cache[query]
        self.misses += 1
        return None

    def set(self, query: str, embedding: np.ndarray) -> None:
        """Cache an embedding for a query."""
        if query in self.cache:
            self.cache.move_to_end(query)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[query] = embedding

    def stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": f"{hit_rate:.1f}%"}


class BM25Index:
    """Simple BM25 index for keyword-based retrieval."""

    def __init__(self):
        self.documents: List[str] = []
        self.doc_freqs: dict = {}
        self.idf: dict = {}
        self.doc_lens: List[int] = []
        self.avgdl: float = 0
        self.k1: float = 1.5
        self.b: float = 0.75

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        return re.findall(r'\w+', text.lower())

    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the index."""
        for doc in documents:
            tokens = self._tokenize(doc)
            self.documents.append(doc)
            self.doc_lens.append(len(tokens))

            # Update document frequencies
            seen = set()
            for token in tokens:
                if token not in seen:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                    seen.add(token)

        # Recalculate IDF and avgdl
        n = len(self.documents)
        self.avgdl = sum(self.doc_lens) / n if n > 0 else 0
        for token, freq in self.doc_freqs.items():
            self.idf[token] = np.log((n - freq + 0.5) / (freq + 0.5) + 1)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search for documents matching the query."""
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc in enumerate(self.documents):
            doc_tokens = self._tokenize(doc)
            doc_len = self.doc_lens[idx]
            score = 0

            token_counts = {}
            for token in doc_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

            for token in query_tokens:
                if token in token_counts:
                    tf = token_counts[token]
                    idf = self.idf.get(token, 0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * numerator / denominator

            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, path: str) -> None:
        """Save index to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'doc_freqs': self.doc_freqs,
                'idf': self.idf,
                'doc_lens': self.doc_lens,
                'avgdl': self.avgdl
            }, f)
        logger.debug(f"Saved BM25 index with {len(self.documents)} documents")

    def load(self, path: str) -> bool:
        """Load index from disk."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.documents = data['documents']
            self.doc_freqs = data['doc_freqs']
            self.idf = data['idf']
            self.doc_lens = data['doc_lens']
            self.avgdl = data['avgdl']
            logger.debug(f"Loaded BM25 index with {len(self.documents)} documents")
            return True
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}")
            return False


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and FAISS for better accuracy.
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        faiss_index: faiss.IndexFlatL2,
        bm25_index: BM25Index | None = None,
    ):
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index or BM25Index()
        self.embedding_cache = EmbeddingCache(settings.embedding_cache_size) if settings.enable_embedding_cache else None

    def _get_embedding(self, query: str) -> np.ndarray:
        """Get embedding with caching."""
        if self.embedding_cache:
            cached = self.embedding_cache.get(query)
            if cached is not None:
                logger.debug("Embedding cache hit")
                return cached

        embedding = self.embedding_model.encode([query])[0]
        embedding = np.array(embedding, dtype="float32").reshape(1, -1)

        if self.embedding_cache:
            self.embedding_cache.set(query, embedding)

        return embedding

    def _reciprocal_rank_fusion(
        self,
        faiss_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        Higher alpha = more weight to FAISS (semantic).
        """
        scores = {}
        alpha = settings.hybrid_alpha

        # Add FAISS scores (weighted by alpha)
        for rank, (idx, _) in enumerate(faiss_results):
            scores[idx] = scores.get(idx, 0) + alpha * (1 / (k + rank + 1))

        # Add BM25 scores (weighted by 1-alpha)
        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + (1 - alpha) * (1 / (k + rank + 1))

        # Sort by combined score
        combined = [(idx, score) for idx, score in scores.items()]
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined

    def retrieve(self, query: str, top_k: int | None = None) -> List[Tuple[int, float]]:
        """
        Retrieve most relevant chunk indices for a query.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of (chunk_index, score) tuples.
        """
        if top_k is None:
            top_k = settings.retrieval_top_k

        if self.faiss_index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []

        # Get FAISS results
        query_embedding = self._get_embedding(query)
        distances, indices = self.faiss_index.search(query_embedding, top_k)

        faiss_results = [
            (int(idx), 1 / (1 + dist))  # Convert distance to similarity
            for idx, dist in zip(indices[0], distances[0])
            if idx != -1
        ]

        # If hybrid search is disabled, return FAISS results only
        if not settings.use_hybrid_search or not self.bm25_index.documents:
            return faiss_results[:top_k]

        # Get BM25 results
        bm25_results = self.bm25_index.search(query, top_k)

        # Combine with RRF
        combined = self._reciprocal_rank_fusion(faiss_results, bm25_results)
        logger.debug(f"Hybrid search: FAISS={len(faiss_results)}, BM25={len(bm25_results)}, Combined={len(combined)}")

        return combined[:top_k]

    def get_cache_stats(self) -> dict | None:
        """Get embedding cache statistics."""
        if self.embedding_cache:
            return self.embedding_cache.stats()
        return None


def load_chunks() -> List[str]:
    """Load all chunks from disk."""
    chunks = []
    chunk_dir = settings.chunks_folder

    if not os.path.exists(chunk_dir):
        return chunks

    # Get all chunk files sorted by index
    chunk_files = []
    for f in os.listdir(chunk_dir):
        if f.startswith("chunk_") and f.endswith(".txt"):
            try:
                idx = int(f.replace("chunk_", "").replace(".txt", ""))
                chunk_files.append((idx, f))
            except ValueError:
                continue

    chunk_files.sort(key=lambda x: x[0])

    for _, filename in chunk_files:
        path = os.path.join(chunk_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks.append(f.read())
        except IOError as e:
            logger.warning(f"Failed to load chunk {filename}: {e}")

    return chunks


def create_hybrid_retriever(
    embedding_model: SentenceTransformer,
    faiss_index: faiss.IndexFlatL2
) -> HybridRetriever:
    """Create and initialize a hybrid retriever."""
    bm25_index = BM25Index()

    # Try to load existing BM25 index
    if not bm25_index.load(settings.bm25_index_path):
        # Build from chunks
        chunks = load_chunks()
        if chunks:
            bm25_index.add_documents(chunks)
            bm25_index.save(settings.bm25_index_path)
            logger.info(f"Built BM25 index with {len(chunks)} chunks")

    return HybridRetriever(embedding_model, faiss_index, bm25_index)
