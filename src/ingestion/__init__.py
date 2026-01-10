"""Ingestion module for document processing with automatic metadata."""

from .document_processor import (
    DocumentProcessor,
    ProcessedChunk,
    DocumentMetadata
)

__all__ = [
    "DocumentProcessor",
    "ProcessedChunk",
    "DocumentMetadata"
]
