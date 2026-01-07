"""Production RAG: semantic chunking + multi-stage LLM filtering."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval import retrieve
from langchain_openai import ChatOpenAI
from config import get_env as env


class ProductionRAG:
    """
    Configuración optimizada para producción:
    - Chunking: semantic (threshold=0.8)
    - Filtering: multi-stage LLM (base → reflect → critic)
    - Retrieval: embeddings + dynamic thresholding
    """

    def __init__(self):
        self.label = "semantic"
        self.use_filtering = True
        self.k = 5
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=env("OPENAI_API_KEY"),
            temperature=0.3
        )

    def query(self, question: str, return_chunks: bool = False):
        """Ejecuta query RAG completo."""
        chunks = retrieve(question, self.label, self.k, self.use_filtering)
        context = "\n\n---\n\n".join(chunks)

        prompt = f"""Responde usando ÚNICAMENTE este contexto:

{context}

Pregunta: {question}
Respuesta:"""

        answer = self.llm.invoke(prompt).content

        if return_chunks:
            return {"answer": answer, "chunks": chunks, "num_chunks": len(chunks)}
        return {"answer": answer}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "¿Cuál es la arquitectura propuesta para Belcorp?"

    rag = ProductionRAG()
    result = rag.query(question, return_chunks=True)

    print(f"\n{'='*60}\nRESPUESTA\n{'='*60}")
    print(result['answer'])
    print(f"\n{'='*60}\nCHUNKS USADOS: {result['num_chunks']}\n{'='*60}")
    for i, chunk in enumerate(result['chunks'], 1):
        preview = chunk[:150].replace('\n', ' ')
        print(f"[{i}] {preview}...")
