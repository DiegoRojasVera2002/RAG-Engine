"""Indexing module for automatic metadata generation."""

from .bm25_keywords import (
    BM25KeywordExtractor,
    KeywordMetadata,
    extract_keywords_from_chunks
)

__all__ = [
    "BM25KeywordExtractor",
    "KeywordMetadata",
    "extract_keywords_from_chunks"
]
