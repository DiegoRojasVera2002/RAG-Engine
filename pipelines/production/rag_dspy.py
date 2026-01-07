"""
Production RAG with DSPy-optimized filtering.
Mantiene las 3 etapas pero con prompts optimizables.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from src.retrieval import retrieve
from langchain_openai import ChatOpenAI
from config import get_env as env
from src.filtering.chunk_filter_dspy import (
    MultiStageChunkScorer,
    filter_chunks_by_relevance_dspy
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ProductionRAGDSPy:
    """
    Production RAG usando DSPy-optimized filtering.

    Configuración:
    - Chunking: semantic (θ=0.8)
    - Filtering: DSPy multi-stage (optimizable)
    - Retrieval: 15 candidatos → 5 filtrados
    """

    def __init__(self, compiled_scorer=None):
        """
        Args:
            compiled_scorer: DSPy scorer pre-compilado/optimizado.
                           Si es None, intenta cargar desde archivo.
        """
        self.label = "semantic"
        self.k = 5

        # Configurar DSPy
        dspy.configure(lm=dspy.LM('openai/gpt-4o-mini', api_key=env("OPENAI_API_KEY")))

        # Cargar scorer compilado si existe
        compiled_path = Path(__file__).parent / "compiled_scorer.json"

        if compiled_scorer:
            logger.info("Using provided compiled scorer")
            self.scorer = compiled_scorer
        elif compiled_path.exists():
            logger.info(f"Loading optimized scorer from {compiled_path}")
            self.scorer = MultiStageChunkScorer()
            self.scorer.load(str(compiled_path))
        else:
            logger.info("Using base DSPy scorer (not optimized)")
            self.scorer = MultiStageChunkScorer()

        # LLM para generación de respuesta
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=env("OPENAI_API_KEY"),
            temperature=0.3
        )

    def query(self, question: str, return_chunks: bool = False):
        """Ejecuta query RAG con DSPy filtering."""
        # Retrieval
        logger.info(f"Retrieving chunks for: {question[:80]}...")
        initial_chunks = self._retrieve_candidates(question)

        # DSPy filtering
        logger.info("Applying DSPy multi-stage filtering...")
        chunks = filter_chunks_by_relevance_dspy(
            self.scorer,
            initial_chunks,
            question,
            min_chunks=self.k
        )
        chunks = chunks[:self.k]

        # Generate answer
        context = "\n\n---\n\n".join(chunks)
        prompt = f"""Responde usando ÚNICAMENTE este contexto:

{context}

Pregunta: {question}
Respuesta:"""

        logger.info("Generating answer...")
        answer = self.llm.invoke(prompt).content

        if return_chunks:
            return {"answer": answer, "chunks": chunks, "num_chunks": len(chunks)}
        return {"answer": answer}

    def _retrieve_candidates(self, question: str):
        """Retrieve initial candidates (sin filtering LLM)."""
        from qdrant_client import QdrantClient
        from langchain_openai import OpenAIEmbeddings

        client = QdrantClient(
            url=env("QDRANT_URL"),
            api_key=env("QDRANT_API_KEY"),
        )
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=env("OPENAI_API_KEY"),
        )

        vector = embeddings.embed_query(question)
        initial_k = self.k * 3  # 15 candidatos

        hits = client.query_points(
            collection_name=f"benchmark_{self.label}",
            query=vector,
            limit=initial_k
        )

        return [h.payload["text"] for h in hits.points]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "¿Cuál es la arquitectura propuesta para Belcorp?"

    print(f"\n{'='*60}\nDSPy RAG PIPELINE\n{'='*60}")
    print(f"Question: {question}\n{'='*60}\n")

    rag = ProductionRAGDSPy()
    result = rag.query(question, return_chunks=True)

    print(f"\n{'='*60}\nRESPUESTA\n{'='*60}")
    print(result['answer'])
    print(f"\n{'='*60}\nCHUNKS USADOS: {result['num_chunks']}\n{'='*60}")
    for i, chunk in enumerate(result['chunks'], 1):
        preview = chunk[:150].replace('\n', ' ')
        print(f"[{i}] {preview}...")
