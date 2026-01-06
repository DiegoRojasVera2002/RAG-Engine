from qdrant_client import QdrantClient
from config import get_env

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=get_env("QDRANT_URL"),
        api_key=get_env("QDRANT_API_KEY"),
    )
