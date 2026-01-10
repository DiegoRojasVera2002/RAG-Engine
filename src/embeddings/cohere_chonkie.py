"""
Cohere Embed v4 wrapper compatible with Chonkie's BaseEmbeddings.

Allows using Cohere Embed v4 from AWS Bedrock with Chonkie's SemanticChunker.
"""

import logging
import numpy as np
from typing import List
from chonkie.embeddings import BaseEmbeddings

from .cohere_bedrock import CohereEmbedV4

logger = logging.getLogger(__name__)


class CohereChonkieEmbeddings(BaseEmbeddings):
    """
    Cohere Embed v4 embeddings handler compatible with Chonkie.

    Implements BaseEmbeddings interface for use with Chonkie's SemanticChunker.
    """

    def __init__(
        self,
        model_id: str = "cohere.embed-v4:0",
        region: str = "us-east-1",
        dimensions: int = 1024
    ):
        """
        Initialize Cohere embeddings for Chonkie.

        Args:
            model_id: Bedrock model ID
            region: AWS region
            dimensions: Embedding dimensions
        """
        self.cohere = CohereEmbedV4(
            model_id=model_id,
            region=region,
            dimensions=dimensions
        )
        self._dimensions = dimensions

        # Initialize tokenizer (uses tiktoken for compatibility)
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except ImportError:
            logger.warning("tiktoken not available, using simple tokenizer")
            self._tokenizer = None

        logger.info(
            f"Initialized Cohere Chonkie Embeddings: {model_id} ({dimensions} dims)"
        )

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimensions

    def embed(self, text: str) -> np.ndarray:
        """
        Embed single text.

        Args:
            text: Input text

        Returns:
            Numpy array of shape (dimension,)
        """
        # Use search_document for chunking (indexing documents)
        return self.cohere.embed(text, input_type="search_document")

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of numpy arrays
        """
        # Cohere supports batch processing (up to 96 texts)
        return self.cohere.embed_batch(texts, input_type="search_document")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken tokenizer.

        Args:
            text: Input text

        Returns:
            Token count
        """
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        else:
            # Fallback: 1 token ≈ 4 characters
            return len(text) // 4

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts."""
        return [self.count_tokens(text) for text in texts]

    def get_tokenizer(self):
        """Get tokenizer (tiktoken cl100k_base)."""
        return self._tokenizer

    @classmethod
    def is_available(cls) -> bool:
        """Check if Cohere embeddings are available."""
        try:
            import boto3
            return True
        except ImportError:
            return False

    def __repr__(self) -> str:
        """String representation."""
        return f"CohereChonkieEmbeddings(model_id='cohere.embed-v4:0', dimensions={self._dimensions})"
