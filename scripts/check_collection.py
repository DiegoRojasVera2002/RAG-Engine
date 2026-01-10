"""
Check script to get information about a Qdrant collection.

This script connects to Qdrant and retrieves details about a specific
collection, such as the number of points it contains.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from qdrant_client import QdrantClient
from config import get_env as env

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Qdrant Configuration
COLLECTION_NAME = "rag_automatic_metadata"

def main():
    """Main function to get and display collection info from Qdrant."""
    logger.info("=" * 80)
    logger.info("Qdrant Collection Status Checker")
    logger.info("=" * 80)
    logger.info(f"  - Target Collection: {COLLECTION_NAME}\n")

    # 1. Initialize Qdrant Client
    try:
        qdrant_client = QdrantClient(
            url=env("QDRANT_URL"),
            api_key=env("QDRANT_API_KEY"),
            timeout=20,
        )
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return

    # 2. Get collection information
    logger.info(f"Fetching details for collection '{COLLECTION_NAME}'...")
    try:
        collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        logger.error(f"Please check if the collection '{COLLECTION_NAME}' exists and the credentials are correct.")
        return

    # 3. Display collection details
    logger.info("Collection Details:")
    logger.info(collection_info)

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
