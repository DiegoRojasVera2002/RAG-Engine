"""
New ingestion pipeline with automatic metadata generation.

Features:
- Document-agnostic processing (PDFs, etc.)
- Automatic RAPTOR hierarchical clustering.
- Automatic BM25 keyword extraction.
- Fully automatic metadata generation, no manual fields needed.
- Embeddings: Cohere Embed v4 via AWS Bedrock.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import get_env as env
from src.ingestion import DocumentProcessor
from src.embeddings import CohereEmbedV4, CohereChonkieEmbeddings

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Qdrant Configuration
COLLECTION_NAME = "rag_automatic_metadata"
DATA_DIRECTORY = "data/Curriculums"

# Embedding Model Configuration
EMBED_MODEL_ID = "cohere.embed-v4:0"
EMBED_DIM = 1536  # Cohere v4 default dimension in Bedrock
INPUT_TYPE = "search_document"
REGION = "us-east-1"

# RAPTOR Clustering Configuration
RAPTOR_LEVELS = 3
RAPTOR_N_CLUSTERS = 10 # Number of clusters at each level, can be adjusted

# BM25 Keyword Configuration
BM25_TOP_K = 10


def setup_qdrant_collection(client: QdrantClient):
    """Deletes and recreates the Qdrant collection."""
    logger.info(f"Recreating Qdrant collection: '{COLLECTION_NAME}'")
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        logger.info("  - Deleted existing collection.")
    except Exception:
        logger.info("  - Collection does not exist, creating new one.")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBED_DIM,
            distance=models.Distance.COSINE,
        ),
    )
    logger.info(f"  - Collection '{COLLECTION_NAME}' created successfully.\n")


def main():
    """Main function to run the automatic ingestion pipeline."""
    logger.info("=" * 80)
    logger.info("STARTING: Automatic Ingestion Pipeline")
    logger.info("=" * 80)
    logger.info(f"  - Data Directory: {DATA_DIRECTORY}")
    logger.info(f"  - Qdrant Collection: {COLLECTION_NAME}")
    logger.info(f"  - Embedding Model: {EMBED_MODEL_ID} ({EMBED_DIM} dims)")
    logger.info(f"  - Metadata: RAPTOR Clustering + BM25 Keywords + Context Engineering\n")

    # 1. Initialize Document Processor
    logger.info("Step 1: Initializing DocumentProcessor...")
    processor = DocumentProcessor(
        # Embeddings config
        model_id=EMBED_MODEL_ID,
        region=REGION,
        dimensions=EMBED_DIM,
        # Clustering config
        n_levels=RAPTOR_LEVELS,
        reduction_factor=RAPTOR_N_CLUSTERS,
        # Keyword config
        top_k_keywords=BM25_TOP_K,
        # Enable Context Engineering (Anthropic Contextual Retrieval)
        enable_contextual_retrieval=True
    )
    logger.info("  - Processor initialized.\n")

    # 2. Process documents
    data_dir = Path(DATA_DIRECTORY)
    logger.info(f"Step 2: Processing all documents from '{data_dir}'...")
    processed_chunks, doc_metadata = processor.process_directory(data_dir)

    if not processed_chunks:
        logger.warning("No chunks were processed. Exiting.")
        return
    logger.info(f"  - Successfully processed {len(doc_metadata)} documents. Total chunks: {len(processed_chunks)}\n")

    # 3. Setup Qdrant
    logger.info("Step 3: Setting up Qdrant collection...")
    qdrant_client = QdrantClient(url=env("QDRANT_URL"), api_key=env("QDRANT_API_KEY"))
    setup_qdrant_collection(qdrant_client)
    
    # 4. Upload to Qdrant
    logger.info(f"Step 4: Uploading {len(processed_chunks)} chunks to Qdrant in batches...")
    
    from qdrant_client.http.models import PointStruct
    import uuid

    points = []
    for chunk in processed_chunks:
        point_id = str(uuid.uuid4())
        payload = chunk.to_dict()
        vector = payload.pop("embedding")
        
        points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        ))

    # Log a sample of the metadata for verification
    if points:
        logger.info("  - Sample payload of the first point:")
        import json
        logger.info(json.dumps(points[0].payload, indent=2, ensure_ascii=False))

    # Upsert points in batches to avoid payload size limits
    BATCH_SIZE = 64
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True
        )
        logger.info(f"  - Upserted batch {i//BATCH_SIZE + 1}/{(len(points) - 1)//BATCH_SIZE + 1}")

    logger.info("  - Upsert operations complete.\n")

    logger.info("=" * 80)
    logger.info("SUCCESS: Automatic Ingestion Pipeline Finished")
    logger.info(f"Processed {len(doc_metadata)} documents, creating {len(processed_chunks)} chunks.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
