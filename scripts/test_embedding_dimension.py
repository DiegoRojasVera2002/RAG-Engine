"""Test script to verify Cohere Bedrock embedding dimensions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import CohereEmbedV4

# Initialize embeddings
embedder = CohereEmbedV4(
    model_id="cohere.embed-v4:0",
    region="us-east-1",
    dimensions=1024
)

# Test with a sample text
test_text = "Hello, this is a test embedding."
embedding = embedder.embed(test_text)

print(f"Expected dimensions: 1024")
print(f"Actual dimensions: {len(embedding)}")
print(f"Embedding shape: {embedding.shape}")
print(f"Match: {len(embedding) == 1024}")
