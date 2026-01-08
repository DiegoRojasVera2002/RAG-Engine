# RAG Engine - ChunkRAG Implementation

Sistema de Retrieval-Augmented Generation (RAG) optimizado con implementación del paper **ChunkRAG** (arXiv:2410.19572v5).

**✨ Características principales:**
- Semantic chunking (θ=0.8)
- Multi-stage LLM filtering con procesamiento paralelo
- **Cohere Rerank v3.5** para reordenamiento óptimo (nuevo)
- Pipeline de producción listo para usar
- 4x más rápido que implementación secuencial
- **DSPy integration** para optimización automática de prompts
- **5 de 7 técnicas del paper ChunkRAG implementadas**

## 📁 Estructura del Proyecto

```
rag-engine/
├── data/                          # PDFs originales
│   ├── Propuesta_Tecnica_Analitica_Avanzada.pdf
│   └── TDD - Learning Journey BCP v2 - 20250429.pdf
│
├── docs/                          # 📚 Documentación
│   └── DSPY_IMPLEMENTATION.md    # Guía completa de DSPy
│
├── pipelines/                     # 🆕 Pipelines de producción
│   └── production/
│       ├── rag.py                # Pipeline original (async filtering)
│       ├── rag_dspy.py           # Pipeline DSPy (prompts optimizables)
│       ├── rag_cohere_rerank.py  # 🆕 Pipeline con Cohere Rerank v3.5
│       ├── rag_only_rerank.py    # Pipeline solo con reranking (comparación)
│       ├── config.py             # Configuración de producción
│       └── compiled_scorer.json  # Scorer DSPy optimizado (generado)
│
├── src/                           # Código fuente base
│   ├── chunking/                  # Estrategias de chunking
│   │   └── semantic_chunk.py     # Semantic chunking (ChunkRAG)
│   │
│   ├── filtering/                 # ChunkRAG multi-stage LLM filtering
│   │   ├── chunk_filter.py       # Base → Self-Reflection → Critic (async)
│   │   ├── chunk_filter_dspy.py  # Versión DSPy optimizable (threads)
│   │   └── reranker.py           # 🆕 Cohere Rerank v3.5 integration
│   │
│   ├── retrieval/                 # Query y recuperación
│   │   └── query.py              # Retrieval con/sin filtering
│   │
│   └── evaluation/                # Evaluación con RAGAS
│       ├── eval_ragas.py         # Pipeline de evaluación
│       └── ground_truth.py       # Respuestas de referencia
│
├── scripts/                       # Scripts de ejecución
│   ├── ingest.py                 # Ingesta de PDFs a Qdrant (semantic only)
│   └── train_dspy.py             # Entrenamiento DSPy con ejemplos
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
# Ejecutar ingesta (crea colección en Qdrant)
uv run scripts/ingest.py
```

Esto crea la colección:
- `benchmark_semantic`: Chunks semánticos (SemanticChunker con threshold=0.8)

*Nota: Chonkie chunking está deshabilitado. Solo se usa semantic chunking para producción.*

### 2. Pipeline de Producción (Recomendado) 🚀

#### Opción A: Pipeline Original (AsyncIO)

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

**Configuración:**
- Chunking: `semantic` (θ=0.8)
- Filtering: `enabled` (multi-stage LLM async)
- Retrieval: 15 candidatos → 5 filtrados
- Velocidad: ~2-3 segundos por query
- Modelo: `gpt-4o-mini`

#### Opción B: Pipeline DSPy (Prompts Optimizables) 🤖

```bash
# Entrenar scorer DSPy (solo una vez, toma 2-5 minutos)
uv run python scripts/train_dspy.py

# Usar pipeline DSPy (carga automáticamente el scorer optimizado)
uv run python pipelines/production/rag_dspy.py "¿Cuál es la arquitectura de Belcorp?"
```

```python
# Desde código Python
from pipelines.production import ProductionRAGDSPy

rag = ProductionRAGDSPy()  # Auto-carga compiled_scorer.json si existe
result = rag.query("¿Qué componentes usa el sistema?", return_chunks=True)

print(result['answer'])
print(f"Chunks usados: {result['num_chunks']}")
```

**Configuración:**
- Chunking: `semantic` (θ=0.8)
- Filtering: `DSPy optimizable` (multi-stage con threads)
- Retrieval: 15 candidatos → 5 filtrados
- Velocidad: ~5-10 segundos por query
- Modelo: `gpt-4o-mini`

#### Opción C: Pipeline con Cohere Rerank v3.5 (Máxima Accuracy) ⭐

```bash
# Pipeline completo con reranking (5 de 7 técnicas del paper)
uv run python pipelines/production/rag_cohere_rerank.py "¿Cuál es la arquitectura de Belcorp?"
```

```python
# Desde código Python
from pipelines.production import CohereRerankRAG

# Configuración completa (recomendada)
rag = CohereRerankRAG(use_filtering=True, use_reranking=True)
result = rag.query("¿Qué componentes usa el sistema?", return_chunks=True)

print(result['answer'])
print(f"Chunks usados: {result['num_chunks']}")
print(f"Filtering: {result['filtering_enabled']}")
print(f"Reranking: {result['reranking_enabled']}")
```

**Configuración:**
- Chunking: `chonkie` (semantic, θ=0.8)
- Filtering: `enabled` (multi-stage LLM async)
- Reranking: `Cohere Rerank v3.5` (AWS Bedrock)
- Pipeline: Vector → Multi-stage filtering → Cohere rerank → Generation
- Retrieval: 15 candidatos → 5 filtrados → 5 reordenados
- Velocidad: ~17-48 segundos por query
- Costo: ~$0.002 por query (solo Cohere)
- Accuracy esperada: **64.9%** (según paper ChunkRAG)
- Modelo: `gpt-4o-mini`

**Requisitos:**
- AWS credentials configuradas (`aws configure`)
- Permisos de Bedrock en región `us-east-1`
- Dependencia: `boto3` (instalar con `uv pip install boto3`)

📚 **Para más detalles sobre DSPy, ver:** [`docs/DSPY_IMPLEMENTATION.md`](docs/DSPY_IMPLEMENTATION.md)

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

### Técnicas Implementadas (5/7 del paper)

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

#### ✅ 5. Cohere Rerank v3.5 (NUEVO)
**Archivo:** `src/filtering/reranker.py`

Basado en el paper (Algorithm 1, líneas 23-24):
- Modelo: `cohere.rerank-v3-5:0` via AWS Bedrock
- Reordena chunks filtrados por relevancia
- Resuelve problema "Lost in the Middle"
- Costo: $2.00 por 1,000 queries (~$0.002 por query)
- Integración: Pipeline `pipelines/production/rag_cohere_rerank.py`

**Formato de request:**
```python
{
  "query": "user query",
  "documents": ["chunk1", "chunk2", ...],
  "top_n": 5,
  "api_version": 2
}
```

### Técnicas No Implementadas (2/7)

- ❌ **Redundancy removal** (similitud >0.9) - Paper sección 2.4
- ❌ **Hybrid retrieval** (BM25 + embeddings) - Paper Algorithm 1, líneas 2-3

*Nota: Con 5/7 técnicas implementadas, el sistema alcanza el 71% de la configuración completa del paper.*

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

## 🤖 DSPy: Optimización Automática de Prompts

**DSPy** (Declarative Self-improving Language Programs) es un framework de Stanford que permite **optimizar prompts automáticamente** en lugar de escribirlos manualmente.

### ¿Por qué DSPy?

El sistema usa **3 prompts diferentes** para filtering multi-etapa (Base, Reflection, Critic). Tradicionalmente:
- ❌ Escribir prompts manualmente
- ❌ Ajustar por prueba y error
- ❌ Difícil de mejorar sistemáticamente

Con DSPy:
- ✅ Prompts optimizables automáticamente
- ✅ Entrenamiento con ejemplos
- ✅ Mejora continua agregando datos

### Diferencia Clave: AsyncIO vs Threads

| Aspecto | Pipeline Original | Pipeline DSPy |
|---------|------------------|---------------|
| Procesamiento | AsyncIO | ThreadPoolExecutor |
| Velocidad | ~2-3s | ~3-4s |
| Prompts | Hardcoded | Optimizables |
| Paralelismo | `asyncio.gather` | `ThreadPoolExecutor(max_workers=15)` |

**¿Por qué threads en DSPy?**

DSPy usa llamadas síncronas a OpenAI internamente, por lo que `asyncio` no funciona. La solución es `ThreadPoolExecutor`:

```python
# Original (AsyncIO)
async def score_chunk(chunk, query):
    response = await llm.ainvoke(prompt)  # Async nativo
    return score

results = await asyncio.gather(*tasks)  # Paralelo

# DSPy (Threads)
def score_chunk_dspy(scorer, chunk, query):
    scores = scorer(chunk=chunk, query=query)  # Sync
    return scores

with ThreadPoolExecutor(max_workers=15) as executor:
    futures = [executor.submit(score_chunk_dspy, ...) for chunk in chunks]
    results = [f.result() for f in as_completed(futures)]  # Paralelo
```

Ambos métodos logran **paralelismo real**, pero con diferentes mecanismos internos.

### Entrenamiento DSPy

```bash
# Entrenar con 8 ejemplos (high/medium/low relevance)
uv run python scripts/train_dspy.py

# Salida:
# - Bootstrapped 4 full traces
# - 2 rondas de optimización
# - Genera: pipelines/production/compiled_scorer.json
```

El archivo `compiled_scorer.json` (10 KB) contiene:
- Prompts optimizados
- Ejemplos few-shot seleccionados
- Configuración del scorer

### Comparación de Resultados

**Mismo query:** "¿Cuál es la arquitectura de Belcorp?"

| Métrica | Pipeline Original | DSPy Optimizado |
|---------|------------------|-----------------|
| Tiempo | 2s | 3s |
| Chunks filtrados | 6 | 7 |
| Threshold | 0.389 | 0.307 |

**Chunks más relevantes (ambos incluyen):**
- Propuesta Técnica Plataforma de Analítica
- Reutilizar activos tecnológicos
- Componentes modulares

**Observación:** DSPy selecciona chunks ligeramente diferentes pero relevantes. Requiere evaluación con RAGAS para validar calidad.

### Próximos Pasos con DSPy

1. **Evaluar con RAGAS**: Comparar métricas vs pipeline original
2. **Ampliar dataset**: Agregar más ejemplos (50-100)
3. **Experimentar con 2 etapas**: Quizás Base + Critic sea suficiente
4. **Optimizadores avanzados**: MIPROv2, COPRO

📚 **Documentación completa:** [`docs/DSPY_IMPLEMENTATION.md`](docs/DSPY_IMPLEMENTATION.md)

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
