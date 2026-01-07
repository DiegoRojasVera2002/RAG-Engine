# RAG Engine - ChunkRAG Implementation

Sistema de Retrieval-Augmented Generation (RAG) optimizado con implementación del paper **ChunkRAG** (arXiv:2410.19572v5).

**✨ Características principales:**
- Semantic chunking (θ=0.8)
- Multi-stage LLM filtering con procesamiento async paralelo
- Pipeline de producción listo para usar
- 4x más rápido que implementación secuencial

## 📁 Estructura del Proyecto

```
rag-engine/
├── data/                          # PDFs originales
│   ├── Propuesta_Tecnica_Analitica_Avanzada.pdf
│   └── TDD - Learning Journey BCP v2 - 20250429.pdf
│
├── pipelines/                     # 🆕 Pipelines de producción
│   └── production/
│       ├── rag.py                # Pipeline optimizado (semantic + async filtering)
│       └── config.py             # Configuración de producción
│
├── src/                           # Código fuente base
│   ├── chunking/                  # Estrategias de chunking
│   │   ├── chonkie_chunk.py      # Token-based (RecursiveChunker)
│   │   └── semantic_chunk.py     # Semantic chunking (ChunkRAG)
│   │
│   ├── filtering/                 # ChunkRAG multi-stage LLM filtering
│   │   └── chunk_filter.py       # Base → Self-Reflection → Critic (async)
│   │
│   ├── retrieval/                 # Query y recuperación
│   │   └── query.py              # Retrieval con/sin filtering
│   │
│   └── evaluation/                # Evaluación con RAGAS
│       ├── eval_ragas.py         # Pipeline de evaluación
│       └── ground_truth.py       # Respuestas de referencia
│
├── scripts/                       # Scripts de ejecución
│   └── ingest.py                 # Ingesta de PDFs a Qdrant
│
├── results/                       # Resultados de evaluaciones
│   ├── baseline/                 # Sin filtering
│   └── filtered/                 # Con ChunkRAG filtering
│
├── config.py                      # Configuración (variables de entorno)
├── qdrant_client_wrapper.py      # Cliente de Qdrant
├── requirements.txt               # Dependencias Python
├── pyproject.toml                 # Configuración del proyecto
└── README.md                      # Este archivo
```

## 🚀 Instalación

```bash
# Instalar dependencias
uv sync

# O con pip
pip install -r requirements.txt
```

## 📝 Configuración

Crear archivo `.env`:

```env
QDRANT_URL=tu_qdrant_url
QDRANT_API_KEY=tu_api_key
OPENAI_API_KEY=tu_openai_key
```

## 🔧 Uso

### 1. Ingesta de Documentos

```bash
# Ejecutar ingesta (crea colecciones en Qdrant)
uv run scripts/ingest.py
```

Esto crea dos colecciones:
- `benchmark_chonkie`: Chunks basados en tokens (RecursiveChunker)
- `benchmark_semantic`: Chunks semánticos (SemanticChunker con threshold=0.8)

### 2. Pipeline de Producción (Recomendado) 🚀

```bash
# Desde terminal
uv run python pipelines/production/rag.py "¿Cuál es la arquitectura de Belcorp?"
```

```python
# Desde código Python
from pipelines.production import ProductionRAG

rag = ProductionRAG()
result = rag.query("¿Qué componentes usa el sistema?", return_chunks=True)

print(result['answer'])
print(f"Chunks usados: {result['num_chunks']}")
```

**Configuración del pipeline de producción:**
- Chunking: `semantic` (θ=0.8)
- Filtering: `enabled` (multi-stage LLM async)
- Retrieval: 15 candidatos → 5 filtrados
- Velocidad: ~10-15 segundos por query
- Modelo: `gpt-4o-mini`

### 3. Evaluación (Benchmarking)

#### Baseline (sin filtering)
```bash
uv run src/evaluation/eval_ragas.py
```

#### Con ChunkRAG Filtering
```bash
uv run src/evaluation/eval_ragas.py --filter
```

Los resultados se guardan en:
- `results/baseline/` - Evaluaciones sin filtering
- `results/filtered/` - Evaluaciones con multi-stage LLM filtering

## 📊 Resultados

### Mejores Configuraciones

**🏆 GANADOR: Semantic Baseline**
```
Context Recall:       0.703
Faithfulness:         0.993
Factual Correctness:  0.398 (60% mejor que Chonkie)
```

### Comparativa Completa

| Configuración | Context Recall | Faithfulness | Factual Correctness |
|---------------|----------------|--------------|---------------------|
| Chonkie Baseline | 1.000 | 0.983 | 0.248 |
| Chonkie Filtered | 1.000 | 1.000 | 0.314 (+26.6%) |
| **Semantic Baseline** | **0.703** | **0.993** | **0.398** 🏆 |
| Semantic Filtered | 0.689 | 1.000 | 0.354 |

### Conclusiones

1. **Semantic Chunking > Token-based**: +60% en precisión factual
2. **Trade-off Recall/Precisión**: Semantic pierde 30% recall pero gana 60% en precisión
3. **Filtering en Semantic**: Contraproducente (-11% factual), los chunks ya están bien filtrados
4. **Filtering en Chonkie**: Beneficioso (+26.6% factual), ayuda a limpiar ruido

## 🧪 Métricas RAGAS

- **Context Recall**: Proporción de información necesaria recuperada
- **Faithfulness**: Fidelidad al contexto (detecta alucinaciones)
- **Factual Correctness**: Precisión factual vs ground truth (F1 score)

## 📚 Implementación ChunkRAG

### Técnicas Implementadas (4/7 del paper)

#### ✅ 1. Semantic Chunking
**Archivo:** `src/chunking/semantic_chunk.py`

Basado en el paper (Sección 3):
- Tokenización por oraciones (NLTK)
- Embeddings: `text-embedding-3-small`
- Threshold de similitud: **θ = 0.8**
- Límite de chunk: **128 tokens (~500 chars)**
- Agrupación por cosine similarity

#### ✅ 2. Multi-stage LLM Filtering (Async Optimizado)
**Archivo:** `src/filtering/chunk_filter.py`

Basado en el paper (Sección 3.2):

1. **Base Score**: LLM evalúa relevancia inicial (0-1)
2. **Self-Reflection**: LLM reflexiona y ajusta score
3. **Critic Evaluation**: Evaluación crítica con heurísticas
4. **Score Final**: `0.3 * base + 0.3 * reflect + 0.4 * critic`

**Optimización async:**
- Procesamiento paralelo de chunks con `asyncio.gather`
- **4x más rápido** que versión secuencial
- Tiempo: ~10-15s vs ~45s

#### ✅ 3. Dynamic Thresholding
**Archivo:** `src/filtering/chunk_filter.py:154-177`

```python
threshold = μ + σ if var < ε else μ
```

Adaptación automática según distribución de scores.

#### ✅ 4. Chunk-level Filtering
**Archivo:** `src/filtering/chunk_filter.py`

Filtrado granular a nivel de chunk (no documento completo).

### Técnicas No Implementadas

- ❌ **Redundancy removal** (similitud >0.9)
- ❌ **Hybrid retrieval** (BM25 + embeddings)
- ❌ **Cohere reranking** (anti "Lost in the middle")

*Nota: El sistema actual funciona bien sin estas optimizaciones adicionales.*

## ⚡ Performance

### Velocidad de Filtering

| Versión | Tiempo (15 chunks) | Optimización |
|---------|-------------------|--------------|
| Secuencial | ~45s | Baseline |
| **Async Paralelo** | **~11s** | **4x más rápido** |

### Métricas de Calidad (RAGAS)

**Pipeline de producción (semantic + filtering):**
- Context Recall: 0.70
- Faithfulness: 0.99
- Factual Correctness: 0.35

## 🔧 Configuración Avanzada

### Desactivar Async Filtering

```python
from src.filtering import filter_chunks_by_relevance

# Usar versión secuencial
chunks = filter_chunks_by_relevance(
    chunks,
    query,
    use_async=False  # Secuencial (legacy)
)
```

### Ajustar Parámetros de Producción

Editar `pipelines/production/config.py`:

```python
ENABLE_FILTERING = True         # Activar/desactivar filtering
INITIAL_RETRIEVAL_K = 15        # Candidatos iniciales
FINAL_CHUNKS_K = 5              # Chunks finales
SEMANTIC_THRESHOLD = 0.8        # Umbral de similitud
```

## 🔗 Referencias

- Paper: [ChunkRAG (arXiv:2410.19572v5)](https://arxiv.org/abs/2410.19572)
- Chonkie: [https://docs.chonkie.ai](https://docs.chonkie.ai)
- RAGAS: [https://docs.ragas.io](https://docs.ragas.io)
- OpenAI Embeddings: [text-embedding-3-large](https://platform.openai.com/docs/guides/embeddings)

## 📄 Licencia

Proyecto educacional/investigación.
