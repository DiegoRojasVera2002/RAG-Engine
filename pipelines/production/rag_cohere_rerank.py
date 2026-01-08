"""Production RAG with Cohere Rerank v3.5 for improved relevance ordering.

Based on ChunkRAG paper (arXiv:2410.19572v5) - Algorithm 1, lines 23-24.
Addresses the "Lost in the Middle" problem using Cohere's rerank-english-v3.5 model.

Pricing: $2.00 per 1,000 queries (up to 100 chunks per query).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval import retrieve
from langchain_openai import ChatOpenAI
from config import get_env as env


class CohereRerankRAG:
    """
    Production RAG with Cohere Rerank v3.5:
    - Chunking: semantic (threshold=0.8)
    - Filtering: multi-stage LLM (base -> reflect -> critic)
    - Reranking: Cohere Rerank v3.5 (Lost-in-Middle mitigation)
    - Retrieval: embeddings + dynamic thresholding + reranking
    """

    def __init__(self, use_filtering: bool = True, use_reranking: bool = True):
        """
        Initialize RAG pipeline with optional filtering and reranking.

        Args:
            use_filtering: Enable multi-stage LLM filtering (ChunkRAG)
            use_reranking: Enable Cohere Rerank v3.5 (requires boto3)
        """
        self.label = "chonkie"
        self.use_filtering = use_filtering
        self.use_reranking = use_reranking
        self.k = 5
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=env("OPENAI_API_KEY"),
            temperature=0.3
        )

    def query(self, question: str, return_chunks: bool = False):
        """
        Execute complete RAG query with optional reranking.

        Pipeline:
        1. Vector retrieval (k*3 candidates if filtering/reranking enabled)
        2. Multi-stage LLM filtering (if use_filtering=True)
        3. Cohere Rerank v3.5 (if use_reranking=True)
        4. LLM response generation

        Args:
            question: User query
            return_chunks: If True, return chunks along with answer

        Returns:
            dict with 'answer' and optionally 'chunks', 'num_chunks'
        """
        chunks = retrieve(
            question,
            self.label,
            self.k,
            use_filtering=self.use_filtering,
            use_reranking=self.use_reranking
        )

        context = "\n\n---\n\n".join(chunks)

        prompt = f"""Responde usando ÚNICAMENTE este contexto:

{context}

Pregunta: {question}
Respuesta:"""

        answer = self.llm.invoke(prompt).content

        if return_chunks:
            return {
                "answer": answer,
                "chunks": chunks,
                "num_chunks": len(chunks),
                "filtering_enabled": self.use_filtering,
                "reranking_enabled": self.use_reranking
            }
        return {"answer": answer}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "¿Cuál es la arquitectura propuesta para Belcorp?"

    # Configure pipeline: filtering=True, reranking=True
    rag = CohereRerankRAG(use_filtering=True, use_reranking=True)
    result = rag.query(question, return_chunks=True)

    print(f"\n{'='*60}\nRESPUESTA\n{'='*60}")
    print(result['answer'])

    print(f"\n{'='*60}\nCONFIGURACION\n{'='*60}")
    print(f"Multi-stage filtering: {'Enabled' if result['filtering_enabled'] else 'Disabled'}")
    print(f"Cohere reranking: {'Enabled' if result['reranking_enabled'] else 'Disabled'}")
    print(f"Chunks usados: {result['num_chunks']}")

    if result['reranking_enabled']:
        cost = result['num_chunks'] / 1000 * 2.00
        print(f"Estimated reranking cost: ${cost:.4f}")

    print(f"\n{'='*60}\nCHUNKS\n{'='*60}")
    for i, chunk in enumerate(result['chunks'], 1):
        preview = chunk[:150].replace('\n', ' ')
        print(f"[{i}] {preview}...")
