"""Comparacion de pipelines: con y sin filtering antes de reranking."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pipelines.production.rag_cohere_rerank import CohereRerankRAG
from pipelines.production.rag_only_rerank import OnlyRerankRAG
import time


def compare():
    question = "¿Cuál es la arquitectura propuesta para Belcorp?"

    print("\n" + "="*80)
    print("COMPARACION DE PIPELINES")
    print("="*80)
    print(f"Query: {question}\n")

    # Pipeline 1: Solo Cohere Reranking
    print("-"*80)
    print("PIPELINE 1: Solo Cohere Reranking (sin filtering)")
    print("-"*80)
    start = time.time()
    rag1 = OnlyRerankRAG()
    result1 = rag1.query(question, return_chunks=True)
    time1 = time.time() - start

    print(f"\nTiempo: {time1:.2f}s")
    print(f"Chunks usados: {result1['num_chunks']}")
    print(f"\nPrimeros 200 chars de respuesta:")
    print(result1['answer'][:200] + "...")

    # Pipeline 2: Multi-Stage Filtering + Cohere Reranking
    print("\n" + "-"*80)
    print("PIPELINE 2: Multi-Stage Filtering + Cohere Reranking")
    print("-"*80)
    start = time.time()
    rag2 = CohereRerankRAG(use_filtering=True, use_reranking=True)
    result2 = rag2.query(question, return_chunks=True)
    time2 = time.time() - start

    print(f"\nTiempo: {time2:.2f}s")
    print(f"Chunks usados: {result2['num_chunks']}")
    print(f"\nPrimeros 200 chars de respuesta:")
    print(result2['answer'][:200] + "...")

    # Comparacion
    print("\n" + "="*80)
    print("ANALISIS")
    print("="*80)
    print(f"\nDiferencia de tiempo: {time2 - time1:.2f}s")
    print(f"Diferencia en numero de chunks: {result2['num_chunks'] - result1['num_chunks']}")

    print("\nVentaja de Pipeline 2 (con filtering):")
    print("- Elimina chunks irrelevantes ANTES de reranking")
    print("- Cohere solo reordena chunks de calidad")
    print("- Contexto mas limpio para el LLM generador")
    print("- Menos tokens enviados al LLM (ahorro de costo)")

    print("\nDesventaja de Pipeline 1 (sin filtering):")
    print("- Cohere reordena TODO (incluyendo basura)")
    print("- El LLM generador recibe chunks de baja calidad")
    print("- Mayor riesgo de respuestas contaminadas")


if __name__ == "__main__":
    compare()
