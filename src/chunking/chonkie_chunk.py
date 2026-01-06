from chonkie import RecursiveChunker
import logging

# Configure logging
logger = logging.getLogger(__name__)


def chunk_text(text: str):
    """
    Perform token-based recursive chunking using Chonkie.

    This is a simple, fast chunker that splits by tokens recursively.
    """
    logger.info("🔄 Starting token-based chunking (RecursiveChunker)...")
    logger.info(f"  └─ Input text: {len(text)} characters")

    chunker = RecursiveChunker()
    chunks = chunker(text)

    # Log statistics
    chunk_texts = [c.text for c in chunks]
    avg_chunk_size = sum(len(ct) for ct in chunk_texts) / len(chunk_texts) if chunk_texts else 0
    avg_tokens = sum(c.token_count for c in chunks) / len(chunks) if chunks else 0

    logger.info(f"✅ Token-based chunking complete!")
    logger.info(f"  └─ Created {len(chunk_texts)} chunks")
    logger.info(f"  └─ Average chunk size: {avg_chunk_size:.1f} chars / {avg_tokens:.1f} tokens")
    logger.info(f"  └─ Size range: {min(len(ct) for ct in chunk_texts) if chunk_texts else 0}-{max(len(ct) for ct in chunk_texts) if chunk_texts else 0} chars")

    return chunk_texts
