import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from config import get_env as env
from src.filtering import filter_chunks_by_relevance
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

def retrieve(query, label, k=5, use_filtering=False):
    """
    Retrieve chunks from vector DB.

    Args:
        query: User query
        label: Chunker label (chonkie/llama)
        k: Number of chunks to retrieve initially
        use_filtering: If True, apply ChunkRAG-style LLM filtering

    Returns:
        List of text chunks
    """
    vector = embeddings.embed_query(query)

    # Retrieve more candidates if filtering is enabled
    initial_k = k * 3 if use_filtering else k

    logging.info(f"Retrieving {initial_k} chunks from {label} collection...")
    hits = client.query_points(
        collection_name=f"benchmark_{label}",
        query=vector,
        limit=initial_k
    )

    chunks = [h.payload["text"] for h in hits.points]
    logging.info(f"Retrieved {len(chunks)} initial chunks")

    if use_filtering and len(chunks) > 0:
        logging.info("Applying ChunkRAG LLM filtering...")
        chunks = filter_chunks_by_relevance(chunks, query, min_chunks=k)
        logging.info(f"After filtering: {len(chunks)} chunks")

    return chunks[:k]  # Return top k after filtering
