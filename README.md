# RAG Engine - ChunkRAG Implementation

Sistema de Retrieval-Augmented Generation (RAG) con implementación completa del paper **ChunkRAG** (arXiv:2410.19572v5).

## 📁 Estructura del Proyecto

```
rag-engine/
├── data/                          # PDFs originales
│   ├── Propuesta_Tecnica_Analitica_Avanzada.pdf
│   └── TDD - Learning Journey BCP v2 - 20250429.pdf
│
├── src/                           # Código fuente principal
│   ├── chunking/                  # Estrategias de chunking
│   │   ├── chonkie_chunk.py      # Token-based (RecursiveChunker)
│   │   └── semantic_chunk.py     # Semantic chunking (ChunkRAG)
│   │
│   ├── filtering/                 # ChunkRAG multi-stage LLM filtering
│   │   └── chunk_filter.py       # Base → Self-Reflection → Critic
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

### 2. Evaluación

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

### Semantic Chunking

Basado en el paper (Sección 3):
- Tokenización por oraciones (NLTK)
- Embeddings: `text-embedding-3-small`
- Threshold de similitud: **θ = 0.8**
- Límite de chunk: **128 tokens (~500 chars)**
- Agrupación por cosine similarity

### Multi-stage LLM Filtering

Basado en el paper (Sección 3.2):

1. **Base Score**: LLM evalúa relevancia inicial (0-1)
2. **Self-Reflection**: LLM reflexiona y ajusta score
3. **Critic Evaluation**: Evaluación crítica con heurísticas
4. **Score Final**: `0.3 * base + 0.3 * reflect + 0.4 * critic`
5. **Dynamic Thresholding**: `threshold = mean + std if var < ε else mean`

## 🔗 Referencias

- Paper: [ChunkRAG (arXiv:2410.19572v5)](https://arxiv.org/abs/2410.19572)
- Chonkie: [https://docs.chonkie.ai](https://docs.chonkie.ai)
- RAGAS: [https://docs.ragas.io](https://docs.ragas.io)

## 📄 Licencia

Proyecto educacional/investigación.
