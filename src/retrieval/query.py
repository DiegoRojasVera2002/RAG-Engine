import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from config import get_env as env
from src.filtering import filter_chunks_by_relevance
from src.filtering.reranker import CohereReranker
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

client = QdrantClient(
    url=env("QDRANT_URL"),
    api_key=env("QDRANT_API_KEY"),
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=env("OPENAI_API_KEY"),
)

def retrieve(query, label, k=5, use_filtering=False, use_reranking=False):
    """
    Retrieve chunks from vector DB with optional filtering and reranking.

    Pipeline:
    1. Vector search (retrieve initial candidates)
    2. [Optional] Multi-stage LLM filtering (ChunkRAG)
    3. [Optional] Cohere Rerank v3.5 (Lost-in-Middle mitigation)

    Args:
        query: User query
        label: Chunker label (chonkie/llama)
        k: Number of chunks to retrieve initially
        use_filtering: If True, apply ChunkRAG-style LLM filtering
        use_reranking: If True, apply Cohere reranking (requires boto3)

    Returns:
        List of text chunks
    """
    vector = embeddings.embed_query(query)

    # Retrieve more candidates if filtering/reranking is enabled
    if use_filtering or use_reranking:
        initial_k = k * 3
    else:
        initial_k = k

    logging.info(f"Retrieving {initial_k} chunks from {label} collection...")
    hits = client.query_points(
        collection_name=f"benchmark_{label}",
        query=vector,
        limit=initial_k
    )

    chunks = [h.payload["text"] for h in hits.points]
    logging.info(f"Retrieved {len(chunks)} initial chunks")

    # Step 1: Multi-stage LLM filtering (optional)
    if use_filtering and len(chunks) > 0:
        logging.info("Applying ChunkRAG LLM filtering...")
        chunks = filter_chunks_by_relevance(chunks, query, min_chunks=k)

        # Si queremos usar secuencialmente aprox 45 segundos
        #chunks = filter_chunks_by_relevance(chunks, query, use_async=False)
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
