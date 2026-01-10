"""
Quick test script for improved BM25 + Context Engineering.
Tests with 3 sample documents.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from src.ingestion import DocumentProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
TEST_DIR = Path("/tmp/test_ingestion")
EMBED_MODEL_ID = "cohere.embed-v4:0"
EMBED_DIM = 1536
REGION = "us-east-1"


def main():
    logger.info("=" * 80)
    logger.info("TESTING: Improved Metadata Pipeline")
    logger.info("=" * 80)
    logger.info(f"  - Test Directory: {TEST_DIR}")
    logger.info(f"  - Model: {EMBED_MODEL_ID}")
    logger.info(f"  - Features: BM25 (improved) + Context Engineering\n")

    # Initialize processor with all improvements
    logger.info("Initializing DocumentProcessor...")
    processor = DocumentProcessor(
        model_id=EMBED_MODEL_ID,
        region=REGION,
        dimensions=EMBED_DIM,
        n_levels=2,
        reduction_factor=10,
        top_k_keywords=10,  # Increased from 5
        enable_contextual_retrieval=True  # NEW: Context Engineering
    )

    # Process test documents
    logger.info("Processing test documents...")
    chunks, metadata = processor.process_directory(TEST_DIR, pattern="*.pdf")

    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total documents: {len(metadata)}")
    logger.info(f"Total chunks: {len(chunks)}")

    # Show sample chunk with improved metadata
    if chunks:
        sample = chunks[0]
        logger.info("\nSample Chunk #1:")
        logger.info(f"  Source: {sample.source}")
        logger.info(f"  Text: {sample.text[:100]}...")
        logger.info(f"\n  🌳 RAPTOR Metadata:")
        logger.info(f"    - cluster_id: {sample.cluster_id}")
        logger.info(f"    - tree_level: {sample.tree_level}")
        logger.info(f"    - cluster_size: {sample.cluster_size}")
        logger.info(f"\n  🔍 BM25 Keywords (Improved):")
        logger.info(f"    - keywords: {sample.keywords}")
        logger.info(f"    - top_keyword: {sample.top_keyword}")
        logger.info(f"    - diversity: {sample.keyword_diversity:.3f}")
        logger.info(f"\n  📝 Context Engineering:")
        logger.info(f"    - context_prefix: {sample.context_prefix}")

    logger.info("\n" + "=" * 80)
    logger.info("Test Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
