"""RAG con SOLO Cohere Reranking (sin Multi-Stage Filtering).
Para comparar contra la version completa."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval import retrieve
from langchain_openai import ChatOpenAI
from config import get_env as env


class OnlyRerankRAG:
    """Solo Cohere Reranking, sin Multi-Stage Filtering."""

    def __init__(self):
        self.label = "chonkie"
        self.use_filtering = False   # Deshabilitado
        self.use_reranking = True    # Solo reranking
        self.k = 5
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=env("OPENAI_API_KEY"),
            temperature=0.3
        )

    def query(self, question: str, return_chunks: bool = False):
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
                "num_chunks": len(chunks)
            }
        return {"answer": answer}


if __name__ == "__main__":
    question = "¿Cuál es la arquitectura propuesta para Belcorp?"

    print("\n" + "="*60)
    print("COMPARACION: Solo Cohere Rerank (sin filtering)")
    print("="*60)

    rag = OnlyRerankRAG()
    result = rag.query(question, return_chunks=True)

    print(f"\n{'='*60}\nRESPUESTA\n{'='*60}")
    print(result['answer'])

    print(f"\n{'='*60}\nCHUNKS USADOS: {result['num_chunks']}\n{'='*60}")
    for i, chunk in enumerate(result['chunks'], 1):
        preview = chunk[:150].replace('\n', ' ')
        print(f"[{i}] {preview}...")
