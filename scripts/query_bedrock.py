"""
Query a Qdrant usando Amazon Titan Embeddings V2.

Ejemplo de uso:
    python scripts/query_bedrock.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import boto3
from qdrant_client import QdrantClient
from config import get_env as env
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

COLLECTION = "benchmark_bedrock"
MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024
REGION = "us-east-1"


def embed_query(query: str) -> list[float]:
    """
    Genera embedding para una query usando Titan V2.

    Args:
        query: Texto de la pregunta

    Returns:
        Vector de embedding (1024 dimensiones)
    """
    client = boto3.client('bedrock-runtime', region_name=REGION)

    request_body = {
        "inputText": query,
        "dimensions": EMBED_DIM,
        "normalize": True
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(request_body)
    )

    result = json.loads(response['body'].read())
    return result['embedding']


def search(query: str, k: int = 5) -> list[dict]:
    """
    Busca chunks relevantes en Qdrant.

    Args:
        query: Pregunta del usuario
        k: Número de chunks a recuperar

    Returns:
        Lista de chunks con scores
    """
    logger.info(f"Query: {query}")
    logger.info(f"Retrieving top {k} chunks from {COLLECTION}...")

    # Generar embedding de la query
    logger.info("  Generating query embedding with Bedrock...")
    query_vector = embed_query(query)
    logger.info(f"  Query vector: {len(query_vector)} dims")

    # Buscar en Qdrant
    qdrant_client = QdrantClient(
        url=env("QDRANT_URL"),
        api_key=env("QDRANT_API_KEY"),
    )

    results = qdrant_client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=k,
        with_payload=True
    ).points

    logger.info(f"  Retrieved {len(results)} chunks\n")

    chunks = []
    for i, hit in enumerate(results, 1):
        chunks.append({
            "rank": i,
            "score": hit.score,
            "text": hit.payload["text"],
            "source": hit.payload["source"]
        })

    return chunks


if __name__ == "__main__":
    # Pregunta de ejemplo
    question = "¿Cuál es la arquitectura propuesta para Belcorp?"

    print("=" * 80)
    print("RAG QUERY - AWS BEDROCK EMBEDDINGS")
    print("=" * 80)
    print(f"Model: {MODEL_ID}")
    print(f"Collection: {COLLECTION}")
    print("=" * 80)
    print()

    # Buscar chunks
    chunks = search(question, k=5)

    # Mostrar resultados
    print("RESULTS")
    print("=" * 80)
    for chunk in chunks:
        print(f"\n[Rank {chunk['rank']}] Score: {chunk['score']:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Text preview: {chunk['text'][:200]}...")
        print("-" * 80)

    print("\nQuery complete!")
