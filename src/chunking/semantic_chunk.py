"""
Semantic Chunking using Chonkie's SemanticChunker.

Based on ChunkRAG paper configuration:
- Threshold: 0.8 (similarity threshold for grouping sentences)
- Chunk size: ~500 characters (approx 128 tokens)
- Semantic grouping based on cosine similarity
"""

from chonkie import SemanticChunker
from chonkie.embeddings import OpenAIEmbeddings
from config import get_env as env
import logging

# Configure logging
logger = logging.getLogger(__name__)


def chunk_text(text: str) -> list[str]:
    """
    Perform semantic chunking using Chonkie's SemanticChunker.

    Configuration matches ChunkRAG paper:
    - threshold=0.8: Similarity threshold for grouping
    - chunk_size=128: Approx 500 characters (paper spec)
    - Uses OpenAI embeddings (text-embedding-3-small)

    Args:
        text: Input document text

    Returns:
        List of semantically coherent text chunks
    """
    logger.info("🔍 Starting semantic chunking (ChunkRAG-style)...")
    logger.info(f"  └─ Input text: {len(text)} characters")

    # Initialize OpenAI embeddings (text-embedding-3-small as per paper)
    logger.info("  └─ Initializing OpenAI embeddings (text-embedding-3-small)...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=env("OPENAI_API_KEY")
    )

    # Create semantic chunker with ChunkRAG paper configuration
    logger.info("  └─ Creating SemanticChunker with config:")
    logger.info("      • threshold=0.8 (similarity threshold)")
    logger.info("      • chunk_size=128 tokens (~500 chars)")
    logger.info("      • similarity_window=3")
    logger.info("      • skip_window=0 (standard semantic grouping)")

    chunker = SemanticChunker(
        embedding_model=embeddings,
        threshold=0.8,           # ChunkRAG paper default
        chunk_size=128,          # ~500 chars (paper spec)
        similarity_window=3,     # Default window for similarity
        min_sentences_per_chunk=1,
        skip_window=0            # No skip-and-merge (standard semantic grouping)
    )

    # Chunk the text
    logger.info("  └─ Computing sentence embeddings and semantic boundaries...")
    chunks = chunker.chunk(text)

    # Log statistics
    chunk_texts = [chunk.text for chunk in chunks]
    avg_chunk_size = sum(len(c) for c in chunk_texts) / len(chunk_texts) if chunk_texts else 0
    avg_tokens = sum(chunk.token_count for chunk in chunks) / len(chunks) if chunks else 0

    logger.info(f"✅ Semantic chunking complete!")
    logger.info(f"  └─ Created {len(chunk_texts)} semantically coherent chunks")
    logger.info(f"  └─ Average chunk size: {avg_chunk_size:.1f} chars / {avg_tokens:.1f} tokens")
    logger.info(f"  └─ Size range: {min(len(c) for c in chunk_texts) if chunk_texts else 0}-{max(len(c) for c in chunk_texts) if chunk_texts else 0} chars")

    # Return as list of strings
    return chunk_texts
