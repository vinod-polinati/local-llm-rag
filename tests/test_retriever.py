"""
Unit tests for the retriever module.
Tests EmbeddingCache and BM25Index functionality.
"""

import os
import tempfile
import numpy as np
import pytest

from retriever import EmbeddingCache, BM25Index


class TestEmbeddingCache:
    """Tests for the EmbeddingCache class."""

    def test_cache_set_and_get(self):
        """Test basic set and get operations."""
        cache = EmbeddingCache(max_size=10)
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        cache.set("test query", embedding)
        result = cache.get("test query")
        
        assert result is not None
        np.testing.assert_array_equal(result, embedding)

    def test_cache_miss_returns_none(self):
        """Test that cache miss returns None."""
        cache = EmbeddingCache(max_size=10)
        result = cache.get("nonexistent query")
        
        assert result is None

    def test_cache_hit_increments_counter(self):
        """Test that cache hits are tracked."""
        cache = EmbeddingCache(max_size=10)
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        cache.set("query", embedding)
        cache.get("query")
        cache.get("query")
        
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 0

    def test_cache_lru_eviction(self):
        """Test that oldest items are evicted when cache is full."""
        cache = EmbeddingCache(max_size=3)
        
        cache.set("query1", np.array([1.0]))
        cache.set("query2", np.array([2.0]))
        cache.set("query3", np.array([3.0]))
        cache.set("query4", np.array([4.0]))  # Should evict query1
        
        assert cache.get("query1") is None
        assert cache.get("query4") is not None

    def test_cache_stats_hit_rate(self):
        """Test hit rate calculation."""
        cache = EmbeddingCache(max_size=10)
        embedding = np.array([1.0])
        
        cache.set("query", embedding)
        cache.get("query")  # hit
        cache.get("query")  # hit
        cache.get("missing")  # miss
        
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert "66.7%" in stats["hit_rate"]


class TestBM25Index:
    """Tests for the BM25Index class."""

    def test_add_documents(self):
        """Test adding documents to the index."""
        bm25 = BM25Index()
        docs = ["Hello world", "Python programming", "Machine learning"]
        
        bm25.add_documents(docs)
        
        assert len(bm25.documents) == 3

    def test_search_returns_results(self):
        """Test that search returns relevant results."""
        bm25 = BM25Index()
        bm25.add_documents([
            "Python is a programming language",
            "Java is also a programming language",
            "Machine learning with Python",
        ])
        
        results = bm25.search("python", top_k=2)
        
        assert len(results) == 2
        # Python-related docs should score higher
        assert results[0][0] in [0, 2]

    def test_search_empty_index(self):
        """Test searching an empty index."""
        bm25 = BM25Index()
        results = bm25.search("query", top_k=5)
        
        assert results == []

    def test_save_and_load(self, temp_dir):
        """Test saving and loading the index."""
        bm25 = BM25Index()
        bm25.add_documents(["Document one", "Document two"])
        
        path = os.path.join(temp_dir, "test_bm25.pkl")
        bm25.save(path)
        
        # Load into new instance
        bm25_loaded = BM25Index()
        success = bm25_loaded.load(path)
        
        assert success
        assert len(bm25_loaded.documents) == 2

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        bm25 = BM25Index()
        success = bm25.load("/nonexistent/path.pkl")
        
        assert success is False

    def test_tokenization(self):
        """Test that tokenization works correctly."""
        bm25 = BM25Index()
        tokens = bm25._tokenize("Hello, World! This is a TEST.")
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "," not in tokens
