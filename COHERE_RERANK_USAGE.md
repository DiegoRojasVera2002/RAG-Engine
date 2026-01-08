# Cohere Rerank v3.5 - Guía de Uso

## Integración Completada

Se ha integrado **Cohere Rerank v3.5** en el pipeline RAG siguiendo el paper ChunkRAG (arXiv:2410.19572v5).

### Archivos Creados/Modificados

1. **`src/filtering/reranker.py`** - Módulo de reranking con Cohere
2. **`src/retrieval/query.py`** - Actualizado con soporte para reranking
3. **`pipelines/production/rag_cohere_rerank.py`** - Pipeline completo con reranking

---

## Comandos de Uso

### 1. RAG Básico (sin reranking)

```bash
python pipelines/production/rag.py "Your question here"
```

**Pipeline:**
- Vector retrieval
- Multi-stage LLM filtering
- Response generation

---

### 2. RAG con Cohere Reranking

```bash
python pipelines/production/rag_cohere_rerank.py "Your question here"
```

**Pipeline:**
- Vector retrieval (k*3 candidates)
- Multi-stage LLM filtering
- **Cohere Rerank v3.5** (ordena por relevancia)
- Response generation

---

## Configuración Avanzada

### Editar Parámetros en el Código

```python
from pipelines.production.rag_cohere_rerank import CohereRerankRAG

# Solo reranking (sin filtering)
rag = CohereRerankRAG(use_filtering=False, use_reranking=True)

# Solo filtering (sin reranking)
rag = CohereRerankRAG(use_filtering=True, use_reranking=False)

# Ambos habilitados (recomendado)
rag = CohereRerankRAG(use_filtering=True, use_reranking=True)

result = rag.query("Your question")
print(result['answer'])
```

---

## Pricing de Cohere Rerank

| Concepto | Costo |
|----------|-------|
| **Precio base** | $2.00 por 1,000 queries |
| **Query** | Hasta 100 document chunks |
| **Ejemplo 1** | 5 chunks = $0.002 |
| **Ejemplo 2** | 100 chunks = $0.002 |
| **Ejemplo 3** | 350 chunks = $0.008 (4 queries) |

**Nota:** Si tu query tiene >100 chunks, se divide en múltiples queries automáticamente.

---

## Ejemplos de Uso

### Ejemplo 1: Query Simple

```bash
source .venv/bin/activate
python pipelines/production/rag_cohere_rerank.py "What is semantic chunking?"
```

**Output esperado:**
```
============================================================
RESPUESTA
============================================================
Semantic chunking is a technique that groups consecutive
sentences based on their semantic similarity...

============================================================
CONFIGURACION
============================================================
Multi-stage filtering: Enabled
Cohere reranking: Enabled
Chunks usados: 5
Estimated reranking cost: $0.0020

============================================================
CHUNKS
============================================================
[1] Semantic chunking serves as the foundational step...
[2] Consecutive sentences are grouped into chunks...
[3] Each chunk is represented using embeddings...
[4] The threshold θ = 0.8 determines when to split...
[5] This approach improves retrieval precision...
```

---

### Ejemplo 2: Comparación con/sin Reranking

```bash
# Sin reranking
python pipelines/production/rag.py "What is the capital of France?"

# Con reranking
python pipelines/production/rag_cohere_rerank.py "What is the capital of France?"
```

**Diferencia esperada:**
- Sin reranking: Chunks en orden de similitud de embeddings
- Con reranking: Chunks reordenados por Cohere para maximizar relevancia

---

## Test del Reranker

Puedes probar el reranker de forma aislada:

```bash
source .venv/bin/activate
python src/filtering/reranker.py
```

Esto ejecuta un test con documentos de ejemplo y muestra los scores de relevancia.

---

## Troubleshooting

### Error: "No module named 'boto3'"

```bash
source .venv/bin/activate
uv pip install boto3
```

### Error: "AccessDeniedException"

El modelo se habilita automáticamente en el primer uso. Espera 2-3 minutos y vuelve a intentar.

### Error: "Malformed input request"

Verifica que tu versión de boto3 sea reciente:

```bash
uv pip install --upgrade boto3
```

### Error: Región no soportada

Cohere Rerank v3.5 está disponible en:
- us-east-1 (Virginia)
- us-west-2 (Oregon)
- ca-central-1 (Canadá)
- eu-central-1 (Frankfurt)
- ap-northeast-1 (Tokio)

Cambia la región en `src/filtering/reranker.py` si es necesario.

---

## Estructura del Pipeline Completo

```
User Query
    |
    v
[1] Vector Retrieval
    | (retrieve k*3 candidates from Qdrant)
    v
[2] Multi-Stage LLM Filtering (opcional)
    | (base score -> self-reflect -> critic)
    | (dynamic thresholding)
    v
[3] Cohere Rerank v3.5 (opcional)
    | (reorder by relevance score)
    v
[4] LLM Response Generation
    | (GPT-4o-mini)
    v
Final Answer
```

---

## Referencia del Paper

**Paper:** ChunkRAG: A Novel LLM-Chunk Filtering Method for RAG Systems
**arXiv:** 2410.19572v5
**Sección:** Algorithm 1, líneas 23-24 (Lost-in-Middle Reranking)

```
23: // Lost-in-Middle Reranking
24: C_final ← Cohere_Rerank(C_threshold, q_rewritten)
```

**Modelo utilizado:** `cohere.rerank-v3-5:0` en AWS Bedrock

---

## Comparación de Métodos

| Método | Multi-Stage Filtering | Cohere Reranking | Accuracy (PopQA) |
|--------|----------------------|------------------|------------------|
| Standard RAG | No | No | 52.8% |
| ChunkRAG (filtering) | Yes | No | 60-62% |
| **ChunkRAG (full)** | **Yes** | **Yes** | **64.9%** |

**Fuente:** Table 1 del paper ChunkRAG

---

## Próximos Pasos

Si quieres implementar las técnicas restantes del paper:

1. **Redundancy Removal** - Eliminar chunks con similitud >0.9
   - Archivo sugerido: `src/filtering/redundancy.py`
   - Referencia: Paper sección 2.4, Algorithm 1 líneas 4-10

2. **Hybrid Retrieval (BM25 + Embeddings)** - Combinar keyword search con semantic search
   - Archivo sugerido: `src/retrieval/hybrid.py`
   - Referencia: Paper página 4, Algorithm 1 líneas 2-3

3. **Query Rewriting** - Optimizar queries para mejor retrieval
   - Archivo sugerido: `src/retrieval/query_rewriter.py`
   - Referencia: Paper sección 2.3

---

## Soporte

Si encuentras problemas o tienes preguntas:

1. Revisa los logs (el sistema usa logging detallado)
2. Verifica las credenciales de AWS (`aws sts get-caller-identity`)
3. Asegúrate de que boto3 esté instalado en el venv
4. Revisa que tu cuenta AWS tenga acceso a Bedrock en us-east-1
