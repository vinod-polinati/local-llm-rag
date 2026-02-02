"""
Pytest configuration and fixtures for the M.I.K.E RAG system tests.
"""

import os
import sys
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_text():
    """Sample text for chunking tests."""
    return """
    Artificial intelligence is transforming the world. Machine learning models can now understand natural language.
    Deep learning has revolutionized computer vision. Neural networks learn patterns from data.
    The future of AI looks promising. Many industries are adopting AI solutions.
    """


@pytest.fixture
def sample_chunks():
    """Sample document chunks for retrieval tests."""
    return [
        "Python is a programming language used for web development and data science.",
        "Machine learning algorithms can classify images and recognize speech.",
        "Natural language processing enables computers to understand human text.",
        "Docker containers provide consistent deployment environments.",
        "Unit testing ensures code quality and prevents regressions.",
    ]
