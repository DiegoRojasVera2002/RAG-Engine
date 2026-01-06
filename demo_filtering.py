"""
Demostración clara de la diferencia entre BASELINE y FILTERING
"""

from query import retrieve
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from config import get_env as env

client = QdrantClient(
    url=env("QDRANT_URL"),
    api_key=env("QDRANT_API_KEY"),
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=env("OPENAI_API_KEY"),
)

# Pregunta de prueba
query = "¿Cuál es la fórmula de ranking utilizada para puntuar los cursos en Learning Journey?"

print("\n" + "="*80)
print("DEMOSTRACIÓN: BASELINE vs FILTERING")
print("="*80)
print(f"\nPregunta: {query}\n")

# Mostrar los top 15 chunks por embedding similarity
vector = embeddings.embed_query(query)
hits = client.query_points(
    collection_name="benchmark_chonkie",
    query=vector,
    limit=15
)

print("\n" + "-"*80)
print("PASO 1: Recuperación por embeddings (top 15 por similitud)")
print("-"*80)
for i, hit in enumerate(hits.points, 1):
    chunk_preview = hit.payload["text"][:150].replace("\n", " ")
    print(f"{i}. Score: {hit.score:.4f} | {chunk_preview}...")

# BASELINE: Top 5 sin filtrar
print("\n\n" + "="*80)
print("BASELINE (Sin Filtrado)")
print("="*80)
print("Toma los top 5 chunks directamente por embedding similarity")
chunks_baseline = retrieve(query, "chonkie", k=5, use_filtering=False)
print(f"\nChunks seleccionados: {len(chunks_baseline)}")
for i, chunk in enumerate(chunks_baseline, 1):
    print(f"\n[{i}] ({len(chunk)} chars)")
    print(chunk[:200].replace("\n", " ") + "...")

# FILTERING: Evalúa 15, filtra, regresa top 5
print("\n\n" + "="*80)
print("WITH FILTERING (ChunkRAG)")
print("="*80)
print("1. Recupera top 15 por embeddings")
print("2. Evalúa cada chunk con LLM (3 etapas: base → reflect → critic)")
print("3. Aplica threshold dinámico")
print("4. Regresa los top 5 MÁS RELEVANTES")
print("\nEvaluando chunks con LLM...")
chunks_filtered = retrieve(query, "chonkie", k=5, use_filtering=True)
print(f"\nChunks seleccionados después de filtrado: {len(chunks_filtered)}")
for i, chunk in enumerate(chunks_filtered, 1):
    print(f"\n[{i}] ({len(chunk)} chars)")
    print(chunk[:200].replace("\n", " ") + "...")

print("\n\n" + "="*80)
print("RESUMEN")
print("="*80)
print("BASELINE:")
print("  - Toma los 5 chunks con mayor similitud de embeddings")
print("  - NO verifica si realmente responden la pregunta")
print("\nFILTERING (ChunkRAG):")
print("  - Evalúa 15 candidatos con LLM")
print("  - Filtra los que NO son relevantes")
print("  - Regresa solo los 5 que SÍ ayudan a responder")
print("\nBENEFICIO:")
print("  - Mayor context_precision en RAGAS")
print("  - El LLM recibe chunks más relevantes")
print("  - Mejor calidad de respuesta final")
print("="*80 + "\n")
