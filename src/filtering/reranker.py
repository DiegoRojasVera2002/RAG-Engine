"""
Cohere Rerank v3.5 integration for AWS Bedrock
Based on ChunkRAG paper (arXiv:2410.19572v5) - Section 3, Algorithm 1 lines 23-24

Addresses the "Lost in the Middle" problem where relevant information
in the middle of long documents tends to be underemphasized.

Pricing: $2.00 per 1,000 queries (up to 100 chunks per query)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging
from typing import List, Dict, Tuple

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    raise ImportError(
        "boto3 is required for Cohere reranking. Install with: uv pip install boto3"
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


class CohereReranker:
    """
    Cohere Rerank v3.5 reranker using AWS Bedrock.

    Solves the "Lost in the Middle" problem by re-evaluating chunks
    with emphasis on contextual centrality.
    """

    def __init__(self, region: str = 'us-east-1', model_id: str = 'cohere.rerank-v3-5:0'):
        """
        Initialize Cohere reranker.

        Args:
            region: AWS region (default: us-east-1)
            model_id: Bedrock model ID (default: cohere.rerank-v3-5:0)
        """
        self.client = boto3.client('bedrock-runtime', region_name=region)
        self.model_id = model_id
        logging.info(f"Initialized CohereReranker with model {model_id} in {region}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 10,
        return_documents: bool = False
    ) -> List[Dict]:
        """
        Rerank documents using Cohere Rerank v3.5.

        Args:
            query: User query
            documents: List of text chunks to rerank
            top_n: Number of top results to return (max: len(documents))
            return_documents: If True, include document text in response

        Returns:
            List of dicts with 'index', 'relevance_score', and optionally 'text'
            Ordered by relevance_score (descending)

        Raises:
            ValueError: If documents list is empty
            ClientError: If Bedrock API call fails
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        # Cohere Rerank supports up to 1000 documents per query
        # If more than 100, it counts as multiple queries for billing
        if len(documents) > 1000:
            logging.warning(
                f"Received {len(documents)} documents. "
                f"Cohere Rerank supports max 1000. Truncating."
            )
            documents = documents[:1000]

        top_n = min(top_n, len(documents))

        # Build request body (return_documents not supported in Bedrock)
        request_body = {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "api_version": 2  # Required for Cohere Rerank v3.5
        }

        logging.info(f"Reranking {len(documents)} documents, returning top {top_n}...")

        try:
            # Call Bedrock API
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )

            # Parse response
            result = json.loads(response['body'].read())

            # Extract results
            reranked = result.get('results', [])

            # Always add original text if requested (Bedrock doesn't return it)
            if return_documents and reranked:
                for item in reranked:
                    idx = item.get('index')
                    if idx is not None and idx < len(documents):
                        item['text'] = documents[idx]

            logging.info(f"Reranked {len(documents)} → {len(reranked)} top results")

            # Log top 3 scores for debugging
            if reranked:
                top_3 = reranked[:3]
                scores_str = ", ".join([f"{r['relevance_score']:.3f}" for r in top_3])
                logging.info(f"Top 3 scores: {scores_str}")

            return reranked

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']

            logging.error(f"Bedrock API error: {error_code} - {error_msg}")

            if error_code == 'AccessDeniedException':
                logging.error(
                    "Model access denied. The model should auto-enable on first use. "
                    "Wait 2-3 minutes and try again."
                )

            raise

    def rerank_and_filter(
        self,
        query: str,
        chunks: List[str],
        top_n: int = 5,
        score_threshold: float = 0.0
    ) -> List[str]:
        """
        Rerank chunks and return only the top N.

        Args:
            query: User query
            chunks: List of text chunks
            top_n: Number of top chunks to return
            score_threshold: Minimum relevance score (0-1). Chunks below this are filtered.

        Returns:
            List of reranked chunks (top N, ordered by relevance)
        """
        if not chunks:
            return []

        # Rerank
        results = self.rerank(query, chunks, top_n=top_n, return_documents=False)

        # Filter by score threshold
        if score_threshold > 0:
            results = [r for r in results if r['relevance_score'] >= score_threshold]
            logging.info(
                f"Filtered {len(results)} chunks with score >= {score_threshold:.3f}"
            )

        # Extract chunks in relevance order
        reranked_chunks = [chunks[r['index']] for r in results]

        return reranked_chunks

    def rerank_with_scores(
        self,
        query: str,
        chunks: List[str],
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Rerank chunks and return them with their relevance scores.

        Args:
            query: User query
            chunks: List of text chunks
            top_n: Number of top chunks to return

        Returns:
            List of tuples (chunk_text, relevance_score)
        """
        if not chunks:
            return []

        results = self.rerank(query, chunks, top_n=top_n, return_documents=False)

        return [
            (chunks[r['index']], r['relevance_score'])
            for r in results
        ]


def rerank_chunks(
    query: str,
    chunks: List[str],
    top_n: int = 5,
    region: str = 'us-east-1'
) -> List[str]:
    """
    Convenience function for quick reranking.

    Args:
        query: User query
        chunks: List of text chunks
        top_n: Number of top chunks to return
        region: AWS region

    Returns:
        List of reranked chunks
    """
    reranker = CohereReranker(region=region)
    return reranker.rerank_and_filter(query, chunks, top_n=top_n)


# Example usage
if __name__ == "__main__":
    # Test the reranker
    test_query = "What is RAG and how does it work?"

    test_docs = [
        "RAG (Retrieval Augmented Generation) combines retrieval with generation to enhance LLM accuracy",
        "Convolutional Neural Networks are used primarily in computer vision tasks",
        "Retrieval Augmented Generation retrieves relevant documents before generating responses",
        "Deep learning models require extensive training data and computational resources",
        "RAG systems use vector databases to store and retrieve semantic information",
        "Transformers are the foundation of modern large language models",
        "ChunkRAG applies chunk-level filtering to improve RAG precision"
    ]

    print("=" * 80)
    print("🧪 Testing Cohere Rerank v3.5")
    print("=" * 80)
    print(f"\nQuery: {test_query}\n")
    print(f"Documents ({len(test_docs)}):")
    for i, doc in enumerate(test_docs):
        print(f"  [{i}] {doc[:70]}...")

    print("\n" + "-" * 80)
    print("Reranking...")
    print("-" * 80 + "\n")

    try:
        reranker = CohereReranker()
        results = reranker.rerank_with_scores(test_query, test_docs, top_n=5)

        print("✅ Top 5 Results:\n")
        for i, (text, score) in enumerate(results, 1):
            print(f"{i}. Score: {score:.4f}")
            print(f"   Text: {text}\n")

        print("=" * 80)
        print(f"💰 Cost: ~${len(test_docs) / 1000 * 2:.4f} (1 query with {len(test_docs)} docs)")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
