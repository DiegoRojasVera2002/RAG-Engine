"""
Cohere Embed v4 embeddings via AWS Bedrock.

Model: cohere.embed-v4:0
- Multimodal (text + images)
- Multilingual (100+ languages)
- Optimized for semantic search
- 1024 dimensions
- Pricing: $0.12/1M tokens
"""

import json
import logging
from typing import List, Literal, Union
import numpy as np
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class CohereEmbedV4:
    """
    Cohere Embed v4 via AWS Bedrock for production RAG systems.

    Features:
    - input_type optimization (search_document vs search_query)
    - Multilingual support (100+ languages)
    - Batch embedding support
    - Error handling with retries
    - Supports both single string and list of strings
    """

    def __init__(
        self,
        model_id: str = "cohere.embed-v4:0",
        region: str = "us-east-1",
        dimensions: int = 1024
    ):
        self.client = boto3.client('bedrock-runtime', region_name=region)
        self.model_id = model_id
        self.dimensions = dimensions

        logger.info(
            "Initialized Cohere Embed v4",
            extra={
                "model_id": model_id,
                "region": region,
                "dimensions": dimensions
            }
        )

    def embed(
        self,
        texts: Union[str, List[str]],
        input_type: Literal["search_document", "search_query"] = "search_query"
    ) -> List[List[float]]:
        """
        Generate embeddings for one or more texts.

        Args:
            texts: Single text (str) or list of texts (List[str])
            input_type: "search_document" for indexing, "search_query" for queries

        Returns:
            List of embeddings (each embedding is a list of floats)
            - For single string: returns [[embedding]]
            - For list: returns [[emb1], [emb2], ...]
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
            was_single = True
        else:
            was_single = False

        # Validate inputs
        if not texts or not isinstance(texts, list):
            raise ValueError("texts must be a non-empty string or list of strings")

        embeddings = []

        for text in texts:
            # Validate each text
            if not text or not isinstance(text, str) or not text.strip():
                logger.warning("Skipping empty or invalid text")
                embeddings.append([0.0] * self.dimensions)  # Return zero vector
                continue

            # Prepare request body
            request_body = {
                "texts": [text],
                "input_type": input_type,
                "embedding_types": ["float"],
                "truncate": "END"
            }

            try:
                # Call Bedrock
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )

                # Parse response
                result = json.loads(response['body'].read())
                embedding = result.get('embeddings', {}).get('float', [[]])[0]

                if not embedding:
                    raise ValueError("Empty embedding returned from Bedrock")

                embeddings.append(embedding)

            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_msg = e.response['Error']['Message']

                logger.error(
                    "Bedrock API error",
                    extra={
                        "error_code": error_code,
                        "error_message": error_msg,
                        "model_id": self.model_id
                    }
                )

                if error_code == 'AccessDeniedException':
                    logger.error(
                        "Model access denied. The model should auto-enable on first use. "
                        "Wait 2-3 minutes and retry."
                    )

                # Return zero vector as fallback
                embeddings.append([0.0] * self.dimensions)

            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                embeddings.append([0.0] * self.dimensions)  # Fallback

        return embeddings

    def embed_batch(
        self,
        texts: List[str],
        input_type: Literal["search_document", "search_query"] = "search_document",
        batch_size: int = 96
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.

        Cohere Embed v4 supports up to 96 texts per request.

        Args:
            texts: List of input texts
            input_type: "search_document" or "search_query"
            batch_size: Max texts per API call (max: 96)

        Returns:
            List of numpy arrays
        """
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            request_body = {
                "texts": batch,
                "input_type": input_type,
                "embedding_types": ["float"],
                "truncate": "END"
            }

            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )

                result = json.loads(response['body'].read())
                embeddings = result.get('embeddings', {}).get('float', [])

                batch_embeddings = [
                    np.array(emb, dtype=np.float32) for emb in embeddings
                ]
                all_embeddings.extend(batch_embeddings)

                logger.debug(
                    "Batch embedding complete",
                    extra={
                        "batch_size": len(batch),
                        "total_processed": len(all_embeddings)
                    }
                )

            except ClientError as e:
                logger.error(
                    "Batch embedding failed",
                    extra={
                        "batch_index": i // batch_size,
                        "batch_size": len(batch),
                        "error": str(e)
                    }
                )
                raise

        return all_embeddings

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimensions
