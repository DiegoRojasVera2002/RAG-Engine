"""
Query script to inspect data in a Qdrant collection.

This script fetches a few points (vectors and their metadata payloads)
from the specified Qdrant collection and prints them to the console
in a readable format.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from qdrant_client import QdrantClient
from config import get_env as env
import json

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Qdrant Configuration
COLLECTION_NAME = "rag_automatic_metadata"
LIMIT = 5  # Number of records to retrieve

def main():
    """Main function to query and display data from Qdrant."""
    logger.info("=" * 80)
    logger.info("Qdrant Data Inspector")
    logger.info("=" * 80)
    logger.info(f"  - Target Collection: {COLLECTION_NAME}")
    logger.info(f"  - Records to fetch: {LIMIT}\n")

    # 1. Initialize Qdrant Client
    try:
        qdrant_client = QdrantClient(
            url=env("QDRANT_URL"),
            api_key=env("QDRANT_API_KEY"),
            timeout=20,
        )
        # Check connection and collection existence
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if COLLECTION_NAME not in collection_names:
            logger.error(f"Collection '{COLLECTION_NAME}' not found in Qdrant.")
            logger.error(f"Available collections: {collection_names}")
            return
            
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        logger.error("Please ensure your QDRANT_URL and QDRANT_API_KEY are set correctly in your environment.")
        return

    # 2. Fetch data from the collection
    logger.info(f"Fetching the first {LIMIT} records from '{COLLECTION_NAME}'...")
    try:
        records, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=LIMIT,
            with_payload=True,  # Crucial to get the metadata
            with_vectors=True, # Also fetch the embeddings
        )
    except Exception as e:
        logger.error(f"Failed to fetch data from Qdrant: {e}")
        return

    if not records:
        logger.warning("No records found in the collection.")
        return
        
    logger.info(f"Successfully retrieved {len(records)} records.\n")

    # 3. Display the records
    for i, record in enumerate(records):
        logger.info("-" * 80)
        logger.info(f"Record #{i + 1} (Point ID: {record.id})")
        logger.info("-" * 80)
        
        # Display Payload (Metadata)
        payload = record.payload
        if payload:
            # Pretty-print the JSON payload
            logger.info("  [METADATA]:")
            print(json.dumps(payload, indent=4, ensure_ascii=False))
        else:
            logger.info("  [METADATA]: (No payload for this record)")
        
        # Display Vector (Embedding)
        vector = record.vector
        if vector:
            # Show the first few dimensions of the vector to confirm it exists
            vector_preview = vector[:3]
            logger.info(f"\n  [EMBEDDING]: (Vector of {len(vector)} dimensions)")
            logger.info(f"    - Preview: {vector_preview} ...\n")
        else:
            logger.info("  [EMBEDDING]: (No vector for this record)\n")

    logger.info("=" * 80)
    logger.info("Inspection Complete")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
