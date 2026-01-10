"""
Debug script to show what RAPTOR is generating for all levels.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
from src.clustering import RAPTORClustering

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Create synthetic embeddings to demonstrate
np.random.seed(42)
n_chunks = 20
embedding_dim = 128
embeddings = [np.random.randn(embedding_dim) for _ in range(n_chunks)]

# Run RAPTOR clustering with 3 levels
clusterer = RAPTORClustering(n_levels=3, reduction_factor=5)
all_levels = clusterer.cluster_hierarchical(embeddings)

logger.info("=" * 80)
logger.info("RAPTOR Hierarchical Clustering Analysis")
logger.info("=" * 80)
logger.info(f"Input: {n_chunks} chunks with {embedding_dim}-dim embeddings\n")

for level_idx, level_metadata in enumerate(all_levels):
    logger.info(f"📊 LEVEL {level_idx}")
    logger.info(f"   Items in this level: {len(level_metadata)}")

    # Count unique clusters
    unique_clusters = set(m.cluster_id for m in level_metadata)
    logger.info(f"   Unique clusters: {len(unique_clusters)}")
    logger.info(f"   Cluster IDs: {sorted(unique_clusters)}")

    # Show sample metadata
    if level_metadata:
        sample = level_metadata[0]
        logger.info(f"   Sample metadata:")
        logger.info(f"     - cluster_id: {sample.cluster_id}")
        logger.info(f"     - tree_level: {sample.tree_level}")
        logger.info(f"     - cluster_size: {sample.cluster_size}")
        logger.info(f"     - parent_cluster_id: {sample.parent_cluster_id}")

    logger.info("")

logger.info("=" * 80)
logger.info("❌ THE PROBLEM:")
logger.info("=" * 80)
logger.info("✅ Level 0: We have CHUNKS with TEXT + embeddings")
logger.info("   → Can be indexed in Qdrant (text + vector)")
logger.info("")
logger.info("❌ Level 1: We have CENTROIDS (just vectors, NO TEXT)")
logger.info("   → Cannot be indexed without text")
logger.info("   → RAPTOR paper uses LLM to generate summary text")
logger.info("")
logger.info("❌ Level 2: We have CENTROIDS (just vectors, NO TEXT)")
logger.info("   → Cannot be indexed without text")
logger.info("   → RAPTOR paper uses LLM to generate summary text")
logger.info("")
logger.info("=" * 80)
logger.info("SOLUTION (from original paper):")
logger.info("=" * 80)
logger.info("For each cluster at level N:")
logger.info("  1. Collect all texts from chunks in that cluster")
logger.info("  2. Send to LLM: 'Summarize these related chunks: [chunks]'")
logger.info("  3. LLM returns summary text")
logger.info("  4. Generate embedding for summary")
logger.info("  5. Index summary text + embedding at level N+1")
logger.info("")
logger.info("Without LLM → We only have centroids (vectors) → Cannot index")
logger.info("=" * 80)
