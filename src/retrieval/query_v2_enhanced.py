"""
Enhanced retrieval v2 with improved metadata support.

Changes from v1:
- Uses Cohere Embed v4 (Bedrock) instead of OpenAI embeddings
- Targets new collection with enhanced metadata:
  * Improved BM25 keywords (multilingüe, n-gramas)
  * Context Engineering (Anthropic Contextual Retrieval)
  * RAPTOR clustering metadata
- Same filtering/reranking pipeline (multi-stage + Cohere rerank)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient
from config import get_env as env
from src.embeddings import CohereEmbedV4
from src.filtering import filter_chunks_by_relevance
from src.filtering.reranker import CohereReranker
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Configuration
COLLECTION_NAME = "rag_automatic_metadata"  # New collection with enhanced metadata
COHERE_MODEL_ID = "cohere.embed-v4:0"
REGION = "us-east-1"
EMBED_DIM = 1536

# Initialize clients (singleton pattern)
_client = None
_embedder = None


def _get_client():
    """Get or create Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantClient(
            url=env("QDRANT_URL"),
            api_key=env("QDRANT_API_KEY"),
        )
    return _client


def _get_embedder():
    """Get or create Cohere embedder."""
    global _embedder
    if _embedder is None:
        _embedder = CohereEmbedV4(
            model_id=COHERE_MODEL_ID,
            region=REGION
        )
        logging.info(f"Initialized Cohere Embed v4: {COHERE_MODEL_ID} ({EMBED_DIM} dims)")
    return _embedder


def _validate_collection():
    """Validate collection exists and has expected schema."""
    client = _get_client()
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        logging.info(f"Collection '{COLLECTION_NAME}' validated: {collection_info.vectors_count} vectors")
        return True
    except Exception as e:
        logging.error(f"Collection validation failed: {e}")
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found or invalid")


def retrieve(query, k=5, use_filtering=False, use_reranking=False):
    """
    Retrieve chunks from enhanced metadata collection.

    Pipeline:
    1. Vector search with Cohere Embed v4 (retrieve initial candidates)
    2. [Optional] Multi-stage LLM filtering (ChunkRAG)
    3. [Optional] Cohere Rerank v3.5 (Lost-in-Middle mitigation)

    Args:
        query: User query
        k: Number of chunks to retrieve finally
        use_filtering: If True, apply ChunkRAG-style LLM filtering
        use_reranking: If True, apply Cohere reranking (requires boto3)

    Returns:
        List of text chunks
    """
    _validate_collection()  # Validate collection at the start

    client = _get_client()
    embedder = _get_embedder()

    # Generate query embedding with Cohere
    logging.info(f"Generating Cohere embedding for query...")
    vector = embedder.embed([query])[0]

    # Retrieve more candidates if filtering/reranking is enabled
    if use_filtering and use_reranking:
        initial_k = k * 4  # Más candidatos para pipeline completo
    elif use_filtering or use_reranking:
        initial_k = k * 3  # Candidatos para una etapa
    else:
        initial_k = k  # Baseline

    logging.info(f"Retrieving {initial_k} chunks from '{COLLECTION_NAME}'...")
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=initial_k,
        with_payload=True
    )

    chunks = [h.payload["text"] for h in hits.points]
    logging.info(f"Retrieved {len(chunks)} initial chunks")

    # Log sample metadata from first result (for debugging)
    if hits.points and logging.getLogger().level == logging.DEBUG:
        sample = hits.points[0].payload
        logging.debug(f"Sample metadata:")
        logging.debug(f"  - Top keyword: {sample.get('top_keyword')}")
        logging.debug(f"  - Cluster: {sample.get('cluster_id')}")
        logging.debug(f"  - Context prefix: {sample.get('context_prefix', '')[:80]}...")

    # Step 1: Multi-stage LLM filtering (optional)
    if use_filtering and len(chunks) > 0:
        logging.info("Applying ChunkRAG LLM filtering (multi-stage)...")
        chunks = filter_chunks_by_relevance(chunks, query, min_chunks=k)
        logging.info(f"After filtering: {len(chunks)} chunks")

    # Step 2: Cohere Reranking (optional)
    if use_reranking and len(chunks) > 0:
        logging.info("Applying Cohere Rerank v3.5...")
        try:
            reranker = CohereReranker()
            chunks = reranker.rerank_and_filter(query, chunks, top_n=k)
            logging.info(f"After reranking: {len(chunks)} chunks")
        except ImportError:
            logging.warning("boto3 not installed. Skipping reranking. Install with: uv pip install boto3")
        except Exception as e:
            logging.error(f"Reranking failed: {e}. Continuing without reranking.")

    return chunks[:k]  # Return top k after filtering/reranking


def retrieve_with_metadata(query, k=5, use_filtering=False, use_reranking=False):
    """
    Retrieve chunks WITH full metadata (enhanced version).

    Same as retrieve() but returns dict with text + metadata instead of just text.

    Args:
        query: User query
        k: Number of chunks to retrieve
        use_filtering: Apply multi-stage LLM filtering
        use_reranking: Apply Cohere reranking

    Returns:
        List of dicts with keys: text, source, keywords, top_keyword,
        context_prefix, cluster_id, cluster_size, score
    """
    client = _get_client()
    embedder = _get_embedder()

    # Generate query embedding
    logging.info(f"Generating Cohere embedding for query...")
    vector = embedder.embed([query])[0]

    # Retrieve candidates
    if use_filtering and use_reranking:
        initial_k = k * 4  # Más candidatos para pipeline completo
    elif use_filtering or use_reranking:
        initial_k = k * 3  # Candidatos para una etapa
    else:
        initial_k = k  # Baseline

    logging.info(f"Retrieving {initial_k} chunks from '{COLLECTION_NAME}'...")
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=initial_k,
        with_payload=True
    )

    # Extract chunks with metadata
    chunks_with_meta = [
        {
            "text": h.payload["text"],
            "score": h.score,
            "source": h.payload.get("source"),
            "keywords": h.payload.get("keywords", []),
            "top_keyword": h.payload.get("top_keyword"),
            "context_prefix": h.payload.get("context_prefix"),
            "cluster_id": h.payload.get("cluster_id"),
            "cluster_size": h.payload.get("cluster_size"),
            "centroid_similarity": h.payload.get("centroid_similarity")
        }
        for h in hits.points
    ]

    logging.info(f"Retrieved {len(chunks_with_meta)} chunks with metadata")

    # Apply filtering (text-only for now)
    if use_filtering and len(chunks_with_meta) > 0:
        logging.info("Applying ChunkRAG LLM filtering...")
        texts = [c["text"] for c in chunks_with_meta]
        filtered_texts = filter_chunks_by_relevance(texts, query, min_chunks=k)

        # Preservar orden y eficiencia
        if use_filtering and len(chunks_with_meta) > 0:
            texts = [c["text"] for c in chunks_with_meta]
            filtered_texts = filter_chunks_by_relevance(texts, query, min_chunks=k)
            
            # Crear lookup por texto (o mejor: por índice)
            filtered_set = set(filtered_texts)
            chunks_with_meta = [
                c for c in chunks_with_meta if c["text"] in filtered_set
            ]
            
            # Reordenar según filtered_texts si es importante
            text_to_chunk = {c["text"]: c for c in chunks_with_meta}
            chunks_with_meta = [text_to_chunk[t] for t in filtered_texts if t in text_to_chunk]

            logging.info(f"After filtering: {len(chunks_with_meta)} chunks")

    # Apply reranking
    if use_reranking and len(chunks_with_meta) > 0:
        logging.info("Applying Cohere Rerank v3.5...")
        try:
            reranker = CohereReranker()
            texts = [c["text"] for c in chunks_with_meta]
            reranked_texts = reranker.rerank_and_filter(query, texts, top_n=k)

            # Reordenar según reranking (preservar orden del reranker)
            text_to_chunk = {c["text"]: c for c in chunks_with_meta}
            chunks_with_meta = [
                text_to_chunk[t] for t in reranked_texts if t in text_to_chunk
            ]
            logging.info(f"After reranking: {len(chunks_with_meta)} chunks")
        except Exception as e:
            logging.error(f"Reranking failed: {e}. Continuing without reranking.")

    return chunks_with_meta[:k]


# Convenience functions for specific use cases
def retrieve_simple(query, k=5):
    """Simple vector search only (baseline)."""
    return retrieve(query, k, use_filtering=False, use_reranking=False)


def retrieve_with_filtering(query, k=5):
    """Vector search + multi-stage LLM filtering."""
    return retrieve(query, k, use_filtering=True, use_reranking=False)


def retrieve_with_reranking(query, k=5):
    """Vector search + Cohere rerank."""
    return retrieve(query, k, use_filtering=False, use_reranking=True)


def retrieve_full_pipeline(query, k=5):
    """Full pipeline: vector + filtering + reranking."""
    return retrieve(query, k, use_filtering=True, use_reranking=True)


if __name__ == "__main__":
    """Quick test of the retrieval system."""
    import sys

    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        test_query = "¿Quién tiene experiencia con AWS y cloud computing?"

    print(f"\n{'='*80}")
    print(f"TESTING ENHANCED RETRIEVAL V2")
    print(f"{'='*80}")
    print(f"Query: {test_query}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embeddings: Cohere Embed v4 ({EMBED_DIM} dims)")
    print(f"{'='*80}\n")

    # Test 1: Simple retrieval
    print(f"\n{'='*80}")
    print("TEST 1: Simple Vector Search")
    print(f"{'='*80}")
    chunks = retrieve_simple(test_query, k=3)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[{i}] {chunk[:150].replace(chr(10), ' ')}...")

    # Test 2: With metadata
    print(f"\n\n{'='*80}")
    print("TEST 2: Retrieval WITH Metadata")
    print(f"{'='*80}")
    chunks_meta = retrieve_with_metadata(test_query, k=3)
    for i, chunk in enumerate(chunks_meta, 1):
        print(f"\n[{i}] Score: {chunk['score']:.4f}")
        print(f"    Source: {chunk['source']}")
        print(f"    Top keyword: {chunk['top_keyword']}")
        print(f"    Cluster: {chunk['cluster_id']} (size: {chunk['cluster_size']})")
        print(f"    Context: {chunk['context_prefix'][:60]}...")
        print(f"    Text: {chunk['text'][:120].replace(chr(10), ' ')}...")

    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}\n")
