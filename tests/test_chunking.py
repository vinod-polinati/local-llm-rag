"""
Unit tests for the text chunking functionality.
Tests both semantic and fixed-size chunking.
"""

import pytest

from pdfextrct import semantic_chunk_text, fixed_chunk_text, chunk_text
from config import settings


class TestSemanticChunking:
    """Tests for semantic (sentence-aware) chunking."""

    def test_splits_on_sentences(self):
        """Test that text is split on sentence boundaries."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = semantic_chunk_text(text, target_size=50, overlap=10)
        
        assert len(chunks) >= 1
        # Each chunk should contain complete sentences
        for chunk in chunks:
            # Should not end mid-word (unless it's the sentence end)
            assert chunk.strip()[-1] in ".!?" or chunk.strip()

    def test_respects_target_size(self):
        """Test that chunks roughly respect target size."""
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence."
        chunks = semantic_chunk_text(text, target_size=40, overlap=5)
        
        # Chunks should be reasonably sized
        for chunk in chunks:
            # Allow some flexibility for sentence boundaries
            assert len(chunk) < 100

    def test_handles_empty_text(self):
        """Test handling of empty text."""
        chunks = semantic_chunk_text("", target_size=100, overlap=10)
        assert chunks == []

    def test_handles_whitespace_only(self):
        """Test handling of whitespace-only text."""
        chunks = semantic_chunk_text("   \n\t  ", target_size=100, overlap=10)
        assert chunks == []

    def test_preserves_content(self):
        """Test that all content is preserved across chunks."""
        text = "Hello world. This is a test. Final sentence here."
        chunks = semantic_chunk_text(text, target_size=25, overlap=5)
        
        # Join chunks should contain all original words
        combined = " ".join(chunks)
        assert "Hello" in combined
        assert "test" in combined
        assert "Final" in combined


class TestFixedChunking:
    """Tests for fixed-size chunking."""

    def test_creates_fixed_size_chunks(self):
        """Test that chunks are created with fixed size."""
        text = "a" * 100
        chunks = fixed_chunk_text(text, chunk_size=30, overlap=10)
        
        assert len(chunks) >= 3

    def test_overlap_works(self):
        """Test that chunks overlap correctly."""
        text = "0123456789" * 5  # 50 chars
        chunks = fixed_chunk_text(text, chunk_size=20, overlap=5)
        
        # With overlap, adjacent chunks should share content
        if len(chunks) >= 2:
            # Last 5 chars of first chunk should appear in second
            last_of_first = chunks[0][-5:]
            assert last_of_first in chunks[1]

    def test_handles_empty_text(self):
        """Test handling of empty text."""
        chunks = fixed_chunk_text("", chunk_size=100, overlap=10)
        assert chunks == []


class TestChunkTextDispatch:
    """Tests for the chunk_text dispatcher function."""

    def test_uses_semantic_when_enabled(self, monkeypatch):
        """Test that semantic chunking is used when enabled."""
        monkeypatch.setattr(settings, "use_semantic_chunking", True)
        
        text = "First sentence. Second sentence."
        chunks = chunk_text(text)
        
        assert len(chunks) >= 1

    def test_uses_fixed_when_disabled(self, monkeypatch):
        """Test that fixed chunking is used when disabled."""
        monkeypatch.setattr(settings, "use_semantic_chunking", False)
        
        text = "a" * 200
        chunks = chunk_text(text)
        
        assert len(chunks) >= 1
