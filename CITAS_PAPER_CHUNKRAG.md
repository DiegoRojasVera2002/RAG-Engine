# Citas del Paper ChunkRAG por Técnica

**Paper**: ChunkRAG: A Novel LLM-Chunk Filtering Method for RAG Systems
**arXiv**: 2410.19572v5
**Fecha**: 23 Apr 2025

---

## Índice de Técnicas

| # | Técnica | Estado | Página Principal |
|---|---------|--------|------------------|
| 1 | [Semantic Chunking](#1-semantic-chunking) | ✅ Implementada | Página 2-3 |
| 2 | [Multi-Stage Relevance Scoring](#2-multi-stage-relevance-scoring) | ✅ Implementada | Página 3-4 |
| 3 | [Dynamic Thresholding](#3-dynamic-thresholding) | ✅ Implementada | Página 4, Algorithm 1 |
| 4 | [Chunk-Level Filtering](#4-chunk-level-filtering) | ✅ Implementada | Página 1-2 |
| 5 | [Redundancy Removal](#5-redundancy-removal) | ❌ No implementada | Página 2, 4 |
| 6 | [Hybrid Retrieval (BM25 + LLM)](#6-hybrid-retrieval-bm25--llm) | ❌ No implementada | Página 4 |
| 7 | [Cohere Reranking](#7-cohere-reranking) | ❌ No implementada | Página 4 |

---

## TÉCNICAS IMPLEMENTADAS ✅

---

## 1. Semantic Chunking

### 📍 Ubicación en el Paper

#### Sección Principal
- **Página**: 2-3
- **Sección**: "3 Methodology → Semantic Chunking"
- **Líneas**: Primera subsección de Methodology

### 🔖 Citas Textuales a Resaltar

#### Cita 1: Definición del Proceso
**Página 2, Sección 3**
```
"Semantic chunking serves as the foundational step of our methodology,
transforming the input document into semantically meaningful units to
facilitate effective retrieval."
```

#### Cita 2: Los 3 Subprocesos
**Página 3, Sección "Semantic Chunking"**
```
"This stage involves three sub-processes:

• Input Preparation: We begin by tokenizing a document D into sentences
  using NLTK's sent_tokenize function. Each sentence is then assigned an
  embedding vector, generated using a pre-trained embedding model
  (text-embedding-3-small).

• Chunk Formation: Consecutive sentences are grouped into chunks based on
  their semantic similarity, measured by cosine similarity. Specifically,
  if the similarity between consecutive sentences drops below a threshold
  (θ = 0.8), a new chunk is created, as this indicates a shift to a
  different subtopic or theme that warrants its own grouping.

• Chunk Embeddings: Each chunk is represented using the same pre-trained
  embedding model as above. The resultant chunk embeddings are stored in
  a vector database to facilitate efficient retrieval during the query phase."
```

#### Cita 3: Parámetros Específicos
**Página 3**
```
"if the similarity between consecutive sentences drops below a threshold
(θ = 0.8), a new chunk is created"

"Each chunk is also further constrained to be under 500 characters to
enable granular search and prevent oversized chunks"
```

### 📊 Referencias Adicionales

- **Figura**: Figure 2 (página 3) - Muestra "Semantic Chunking" en el pipeline
- **Tabla**: Table 2 (página 7) - "Chunk Analysis Across Similarity Thresholds"

### ✏️ Qué Resaltar

```
PÁGINA 2-3, SECCIÓN "3 METHODOLOGY - Semantic Chunking":
┌─────────────────────────────────────────────────────────┐
│ ✅ Párrafo completo de "Semantic chunking serves..."   │
│ ✅ Los 3 bullets (Input Preparation, Chunk Formation,   │
│    Chunk Embeddings)                                    │
│ ✅ "θ = 0.8" (threshold)                                │
│ ✅ "under 500 characters"                               │
│ ✅ "text-embedding-3-small"                             │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Multi-Stage Relevance Scoring

### 📍 Ubicación en el Paper

#### Sección Principal
- **Página**: 3-4
- **Sección**: "3 Methodology → Hybrid Retrieval and Advanced Filtering → Relevance Scoring and Tresholding"
- **Sub-título**: "Relevance Scoring and Tresholding" (nota: typo en el paper, dice "Tresholding" en vez de "Thresholding")

### 🔖 Citas Textuales a Resaltar

#### Cita 1: Descripción del Proceso Multi-Stage
**Página 4, Sección "Relevance Scoring and Tresholding"**
```
"Each chunk's relevance is evaluated through a multi-stage process:
an LLM assigns initial scores, followed by self-reflection and critic
model refinements. The self-reflection step assesses query alignment,
while the critic applies domain-specific heuristics (e.g., temporal
consistency for time-sensitive queries)."
```

#### Cita 2: Líneas del Algorithm 1
**Página 5, Algorithm 1, líneas 11-17**
```
11: // Multi-stage Scoring
12: for each chunk c ∈ Cfiltered do
13:     base ← LLMRelevance(c, qrewritten)
14:     reflect ← SelfReflect(c, qrewritten, base)
15:     critic ← CriticEval(c, qrewritten, base, reflect)
16:     score(c) ← CombineScores(base, reflect, critic)
17: end for
```

#### Cita 3: Self-Reflection Prompt (Appendix)
**Página 10, Appendix A.1, "Self-Reflection Prompt"**
```
"You have assigned a relevance score to a text chunk based on a user query.
Your initial score was: {score}

Reflect on your scoring and adjust the score if necessary.
Provide the final score."
```

### 📊 Referencias Adicionales

- **Algorithm 1**: Líneas 11-17 (página 5)
- **Figure 2**: "LLM-Based Scoring" box muestra "Initial Score → Self-Reflection → Critic LLM Scoring"
- **Appendix A.1**: Página 10 - Todos los prompts (Relevance Scoring, Self-Reflection)

### ✏️ Qué Resaltar

```
PÁGINA 4, SECCIÓN "Relevance Scoring and Tresholding":
┌─────────────────────────────────────────────────────────┐
│ ✅ "multi-stage process: an LLM assigns initial scores, │
│    followed by self-reflection and critic model         │
│    refinements"                                         │
│ ✅ "self-reflection step assesses query alignment"      │
│ ✅ "critic applies domain-specific heuristics"          │
└─────────────────────────────────────────────────────────┘

PÁGINA 5, ALGORITHM 1, LÍNEAS 11-17:
┌─────────────────────────────────────────────────────────┐
│ ✅ Resaltar todo el bloque "// Multi-stage Scoring"    │
│ ✅ Las 3 funciones: LLMRelevance, SelfReflect,          │
│    CriticEval                                           │
│ ✅ CombineScores (combina los 3 scores)                 │
└─────────────────────────────────────────────────────────┘

PÁGINA 10, APPENDIX A.1:
┌─────────────────────────────────────────────────────────┐
│ ✅ "Relevance Scoring Prompt" completo                  │
│ ✅ "Self-Reflection Prompt" completo                    │
│ ✅ (Opcional) "Critic" no tiene prompt separado en      │
│    appendix, pero se menciona en línea 15 del algoritmo│
└─────────────────────────────────────────────────────────┘
```

---

## 3. Dynamic Thresholding

### 📍 Ubicación en el Paper

#### Sección Principal
- **Página**: 4 (descripción), 5 (algoritmo)
- **Sección**: "3 Methodology → Relevance Scoring and Tresholding"
- **Algorithm 1**: Líneas 18-22

### 🔖 Citas Textuales a Resaltar

#### Cita 1: Descripción del Dynamic Threshold
**Página 4, Sección "Relevance Scoring and Tresholding"**
```
"A dynamic threshold, based on score distribution analysis, determines
final chunk selection. When scores cluster tightly, the threshold
increases to retain only the most relevant chunks."
```

#### Cita 2: Algorithm 1 - Dynamic Thresholding
**Página 5, Algorithm 1, líneas 18-22**
```
18: // Dynamic Thresholding
19: S ← { score(c) | c ∈ Cfiltered}
20: μ ← mean(S); σ ← std(S)
21: T ← if var(S) < ϵ then μ + σ else μ
22: Cthreshold ← { c ∈ Cfiltered | score(c) ≥ T}
```

**ESTA ES LA FÓRMULA CLAVE** ⭐

#### Cita 3: Explicación en Discusión
**Página 7, Sección "7 Discussion"**
```
"The ablation study highlights redundancy filtering's key role in
ChunkRAG, with dynamic chunk merging and optimal similarity thresholds
(validated at θ = 0.8) balancing chunk reduction and relevance while
preventing over-filtering."
```

### 📊 Referencias Adicionales

- **Algorithm 1**: Línea 21 contiene la fórmula del threshold
- **Figure 3**: Página 6 - "Chunk Reduction vs. Similarity Threshold"
- **Table 2**: Página 7 - Muestra valores de threshold 0.5-0.9

### ✏️ Qué Resaltar

```
PÁGINA 4, SECCIÓN "Relevance Scoring and Tresholding":
┌─────────────────────────────────────────────────────────┐
│ ✅ "A dynamic threshold, based on score distribution    │
│    analysis, determines final chunk selection"          │
│ ✅ "When scores cluster tightly, the threshold increases│
│    to retain only the most relevant chunks"             │
└─────────────────────────────────────────────────────────┘

PÁGINA 5, ALGORITHM 1, LÍNEAS 18-22:
┌─────────────────────────────────────────────────────────┐
│ ⭐⭐⭐ RESALTAR CON FLUORESCENTE ⭐⭐⭐                   │
│                                                         │
│ ✅ Línea 21: T ← if var(S) < ϵ then μ + σ else μ       │
│                                                         │
│ Esta es la fórmula matemática del threshold dinámico   │
│ Explicación:                                            │
│   - Si varianza baja: threshold = media + desv_std     │
│   - Si varianza alta: threshold = media                │
└─────────────────────────────────────────────────────────┘

PÁGINA 7, TABLE 2:
┌─────────────────────────────────────────────────────────┐
│ ✅ Toda la tabla "Chunk Analysis Across Similarity      │
│    Thresholds"                                          │
│ ✅ Especialmente θ = 0.8 (el óptimo validado)          │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Chunk-Level Filtering

### 📍 Ubicación en el Paper

#### Sección Principal
- **Página**: 1-2 (introducción y problema), 3-4 (metodología)
- **Sección**: "1 Introduction" y "3 Methodology"

### 🔖 Citas Textuales a Resaltar

#### Cita 1: Definición del Problema (Introducción)
**Página 1-2, Sección "1 Introduction"**
```
"Current RAG systems often retrieve large document segments, assuming
more content means better coverage. However, this overlooks the need
to evaluate smaller sections independently, leading to the inclusion
of irrelevant information."
```

#### Cita 2: Propuesta de ChunkRAG
**Página 2, Sección "1 Introduction"**
```
"We propose ChunkRAG, a novel approach of LLM-driven chunk filtering.
This framework operates at a finer level of granularity than traditional
systems by supporting chunk-level filtering of retrieved information.
Rather than determining the relevance of entire documents, our framework
evaluates both the user query and the individual chunks within the
retrieved chunks."
```

#### Cita 3: Beneficio del Chunk-Level Filtering
**Página 2, Abstract**
```
"The analysis further demonstrates that chunk-level filtering reduces
redundant and weakly related information, enhancing the factual
consistency of responses."
```

#### Cita 4: Comparación con Document-Level
**Página 1, Abstract**
```
"Existing document-level retrieval approaches lack sufficient granularity
to effectively filter non-essential content."
```

### 📊 Referencias Adicionales

- **Figure 1**: Página 1 - Comparación visual "With and Without Chunk Filtering"
- **Section 5.2**: Página 6 - "Insights" explica por qué chunk-level es mejor

### ✏️ Qué Resaltar

```
PÁGINA 1, ABSTRACT:
┌─────────────────────────────────────────────────────────┐
│ ✅ "Existing document-level retrieval approaches lack   │
│    sufficient granularity to effectively filter         │
│    non-essential content"                               │
└─────────────────────────────────────────────────────────┘

PÁGINA 1-2, INTRODUCTION:
┌─────────────────────────────────────────────────────────┐
│ ✅ "Current RAG systems often retrieve large document   │
│    segments, assuming more content means better coverage│
│ ✅ "This framework operates at a finer level of         │
│    granularity than traditional systems by supporting   │
│    chunk-level filtering"                               │
│ ✅ "Rather than determining the relevance of entire     │
│    documents, our framework evaluates... individual     │
│    chunks"                                              │
└─────────────────────────────────────────────────────────┘

PÁGINA 6, SECTION 5.2 "Insights":
┌─────────────────────────────────────────────────────────┐
│ ✅ "chunk-level filtering offers greater benefits in    │
│    short, fact-intensive tasks like PopQA—where even    │
│    minor irrelevant segments can lead to hallucinations"│
└─────────────────────────────────────────────────────────┘

FIGURA 1 (PÁGINA 1):
┌─────────────────────────────────────────────────────────┐
│ ✅ Toda la figura mostrando:                            │
│    - Sin filtering: respuesta con info irrelevante      │
│    - Con LLM chunk filtering: respuesta precisa         │
└─────────────────────────────────────────────────────────┘
```

---

## TÉCNICAS NO IMPLEMENTADAS ❌

---

## 5. Redundancy Removal

### 📍 Ubicación en el Paper

#### Sección Principal
- **Página**: 2 (Related Works), 4 (Methodology)
- **Sección**: "2.4 Redundancy Reduction with Cosine Similarity" y "Initial Filtering"

### 🔖 Citas Textuales a Resaltar

#### Cita 1: Descripción en Related Works
**Página 2, Sección "2.4 Redundancy Reduction with Cosine Similarity"**
```
"Redundant information in retrieved documents can clutter context.
Using cosine similarity, near-identical sections can be deduplicated
by filtering chunks exceeding a similarity threshold (e.g., > 0.9)
(Gan et al., 2024), streamlining input and reducing confusion from
repetition."
```

**THRESHOLD CLAVE**: > 0.9 ⭐

#### Cita 2: Implementación en Methodology
**Página 4, Sección "Initial Filtering"**
```
"Retrieved chunks are initially filtered using a combination of TF-IDF
scoring and cosine similarity. Chunks with high redundancy
(similarity > 0.9) are eliminated."
```

#### Cita 3: En Algorithm 1
**Página 5, Algorithm 1, líneas 4-10**
```
4: // Redundancy Removal
5: Cfiltered ← ∅
6: for each chunk ci ∈ C do
7:     if max cos(emb(ci), emb(cj)) ≤ λdup then
        cj∈Cfiltered
8:         Append ci to Cfiltered
9:     end if
10: end for
```

**Parámetro**: λdup = 0.9 (línea 3: `Require: λdup: Redundancy threshold (e.g., 0.9)`)

### 📊 Referencias Adicionales

- **Algorithm 1**: Líneas 4-10 (página 5)
- **Section 6.1**: Página 6 - "Redundancy Filtering Effectiveness"
- **Figure 3**: Página 6 - Muestra reducción de chunks por threshold

### ✏️ Qué Resaltar

```
PÁGINA 2, SECCIÓN "2.4 Redundancy Reduction with Cosine Similarity":
┌─────────────────────────────────────────────────────────┐
│ ✅ Toda la sección completa (4-5 líneas)                │
│ ⭐ "similarity threshold (e.g., > 0.9)"                 │
│ ✅ "deduplicated by filtering chunks exceeding"         │
└─────────────────────────────────────────────────────────┘

PÁGINA 4, SECCIÓN "Initial Filtering":
┌─────────────────────────────────────────────────────────┐
│ ✅ "Chunks with high redundancy (similarity > 0.9) are  │
│    eliminated"                                          │
└─────────────────────────────────────────────────────────┘

PÁGINA 5, ALGORITHM 1, LÍNEAS 4-10:
┌─────────────────────────────────────────────────────────┐
│ ⭐⭐ RESALTAR TODO EL BLOQUE ⭐⭐                        │
│                                                         │
│ ✅ Línea 3: "λdup: Redundancy threshold (e.g., 0.9)"   │
│ ✅ Líneas 4-10: Algoritmo completo de redundancy removal│
│ ✅ Línea 7: Condición "if max cos(...) ≤ λdup"         │
└─────────────────────────────────────────────────────────┘

PÁGINA 6, FIGURE 3:
┌─────────────────────────────────────────────────────────┐
│ ✅ Gráfica "Chunk Reduction vs Similarity Threshold"    │
│ ✅ Nota que en threshold 0.9 → 8.5% reduction          │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Hybrid Retrieval (BM25 + LLM)

### 📍 Ubicación en el Paper

#### Sección Principal
- **Página**: 4
- **Sección**: "3 Methodology → Hybrid Retrieval and Advanced Filtering → Hybrid Retrieval Strategy"

### 🔖 Citas Textuales a Resaltar

#### Cita 1: Descripción del Hybrid Retrieval
**Página 4, Sección "Hybrid Retrieval Strategy"**
```
"We combine BM25 and LLM-based retrieval methods with equal weights
(0.5 each) to balance keyword and semantic matching."
```

**PESOS CLAVE**: 0.5 BM25 + 0.5 LLM ⭐

#### Cita 2: En Algorithm 1
**Página 5, Algorithm 1, líneas 2-3**
```
2: // Hybrid Retrieval
3: C ← CombineRetrieval(BM25(D, qrewritten), LLM(D, qrewritten), wbm25, wllm)
```

#### Cita 3: Parámetros del Algorithm 1
**Página 5, Algorithm 1, Requirements**
```
Require: wbm25, wllm: Hybrid retrieval weights
```

### 📊 Referencias Adicionales

- **Algorithm 1**: Líneas 2-3 (página 5)
- **Figure 2**: Página 3 - Muestra "Base Retriever" y "BM25 Retriever" con pesos 0.5

### ✏️ Qué Resaltar

```
PÁGINA 4, SECCIÓN "Hybrid Retrieval Strategy":
┌─────────────────────────────────────────────────────────┐
│ ⭐⭐ RESALTAR TODA LA SECCIÓN ⭐⭐                       │
│                                                         │
│ ✅ "We combine BM25 and LLM-based retrieval methods     │
│    with equal weights (0.5 each)"                      │
│ ✅ "to balance keyword and semantic matching"           │
└─────────────────────────────────────────────────────────┘

PÁGINA 5, ALGORITHM 1, LÍNEAS 2-3:
┌─────────────────────────────────────────────────────────┐
│ ✅ Línea 2: "// Hybrid Retrieval"                      │
│ ✅ Línea 3: "CombineRetrieval(BM25(...), LLM(...),     │
│             wbm25, wllm)"                               │
└─────────────────────────────────────────────────────────┘

PÁGINA 3, FIGURE 2:
┌─────────────────────────────────────────────────────────┐
│ ✅ "Base Retriever" box                                 │
│ ✅ "BM25 Retriever" box                                 │
│ ✅ "Weight 0.5" labels en ambos                         │
│ ✅ "Ensemble Retriever" que combina ambos               │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Cohere Reranking

### 📍 Ubicación en el Paper

#### Sección Principal
- **Página**: 4
- **Sección**: "3 Methodology → Hybrid Retrieval and Advanced Filtering → Hybrid Retrieval Strategy"

### 🔖 Citas Textuales a Resaltar

#### Cita 1: Descripción del Reranking
**Página 4, Sección "Hybrid Retrieval Strategy"**
```
"Cohere's reranking model (rerank-english-v3.0) then addresses the
Lost in the middle problem - where relevant information in the middle
of long documents tends to be underemphasized by standard retrieval
methods - by re-evaluating chunks with emphasis on contextual centrality,
preventing the oversight of relevant mid-document information."
```

**MODELO CLAVE**: rerank-english-v3.0 ⭐

#### Cita 2: En Algorithm 1
**Página 5, Algorithm 1, líneas 23-24**
```
23: // Lost-in-Middle Reranking
24: Cfinal ← Cohere_Rerank(Cthreshold, qrewritten)
```

### 📊 Referencias Adicionales

- **Algorithm 1**: Líneas 23-24 (página 5) - último paso antes del return
- **Figure 2**: Página 3 - Muestra "COHERE RE-RANK" como paso final

### ✏️ Qué Resaltar

```
PÁGINA 4, SECCIÓN "Hybrid Retrieval Strategy":
┌─────────────────────────────────────────────────────────┐
│ ⭐⭐ TODO EL PÁRRAFO DE COHERE ⭐⭐                      │
│                                                         │
│ ✅ "Cohere's reranking model (rerank-english-v3.0)"    │
│ ✅ "addresses the Lost in the middle problem"           │
│ ✅ "re-evaluating chunks with emphasis on contextual    │
│    centrality"                                          │
│ ✅ "preventing the oversight of relevant mid-document   │
│    information"                                         │
└─────────────────────────────────────────────────────────┘

PÁGINA 5, ALGORITHM 1, LÍNEAS 23-24:
┌─────────────────────────────────────────────────────────┐
│ ✅ Línea 23: "// Lost-in-Middle Reranking"             │
│ ✅ Línea 24: "Cfinal ← Cohere_Rerank(Cthreshold,       │
│              qrewritten)"                               │
└─────────────────────────────────────────────────────────┘

PÁGINA 3, FIGURE 2:
┌─────────────────────────────────────────────────────────┐
│ ✅ Box final "COHERE RE-RANK"                           │
│ ✅ Es el último paso antes de la respuesta final        │
└─────────────────────────────────────────────────────────┘
```

---

## Resumen de Páginas Clave

### 📚 Tabla Rápida de Referencia

| Página | Contenido Clave |
|--------|-----------------|
| **1** | Abstract, Introducción, Figure 1 (chunk filtering comparison) |
| **2** | Related Works, Redundancy (sección 2.4), Inicio Methodology |
| **3** | Semantic Chunking (detallado), Figure 2 (pipeline completo) |
| **4** | Multi-stage scoring, Dynamic threshold, Hybrid retrieval, Cohere |
| **5** | **Algorithm 1** (⭐ MÁS IMPORTANTE) - Todo el pipeline |
| **6** | Analysis, Figure 3 (redundancy), Figure 4 (similarity) |
| **7** | Table 1 (resultados), Table 2 (thresholds), Discussion |
| **10** | Appendix A.1 - Prompts exactos para LLM scoring |

---

## Mapa Visual del Paper

```
┌─────────────────────────────────────────────────────────────────┐
│ PÁGINA 1: ABSTRACT + INTRODUCTION                               │
├─────────────────────────────────────────────────────────────────┤
│ ✅ Chunk-level filtering (definición del problema)              │
│ ✅ Figure 1 (comparación visual)                                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PÁGINA 2: RELATED WORKS                                         │
├─────────────────────────────────────────────────────────────────┤
│ ❌ Sección 2.4: Redundancy Removal                              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PÁGINA 3: METHODOLOGY - SEMANTIC CHUNKING                       │
├─────────────────────────────────────────────────────────────────┤
│ ✅ Semantic Chunking (3 subprocesos)                            │
│ ✅ θ = 0.8, 500 chars, text-embedding-3-small                  │
│ ✅ Figure 2: Pipeline completo                                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PÁGINA 4: METHODOLOGY - FILTERING & RETRIEVAL                   │
├─────────────────────────────────────────────────────────────────┤
│ ✅ Multi-stage scoring (base, reflect, critic)                  │
│ ✅ Dynamic thresholding (descripción)                           │
│ ❌ Redundancy removal (similarity > 0.9)                        │
│ ❌ Hybrid Retrieval (BM25 0.5 + LLM 0.5)                        │
│ ❌ Cohere rerank-english-v3.0                                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PÁGINA 5: ALGORITHM 1 ⭐⭐⭐ MÁS IMPORTANTE ⭐⭐⭐             │
├─────────────────────────────────────────────────────────────────┤
│ Líneas 1-3:   Hybrid Retrieval (BM25 + LLM)                    │
│ Líneas 4-10:  ❌ Redundancy Removal                             │
│ Líneas 11-17: ✅ Multi-stage Scoring                            │
│ Líneas 18-22: ✅ Dynamic Thresholding (fórmula μ+σ vs μ)       │
│ Líneas 23-24: ❌ Cohere Reranking                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PÁGINA 6-7: RESULTS & ABLATION                                  │
├─────────────────────────────────────────────────────────────────┤
│ Figure 3: Redundancy effectiveness                              │
│ Table 1: Accuracy results (PopQA 64.9%, PubHealth 77.3%)       │
│ Table 2: Threshold analysis (0.5 to 0.9)                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PÁGINA 10: APPENDIX A.1 - PROMPTS                              │
├─────────────────────────────────────────────────────────────────┤
│ ✅ Relevance Scoring Prompt (base score)                        │
│ ✅ Self-Reflection Prompt                                       │
│ ✅ Threshold Determination Prompt                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Guía de Colores para Resaltado

Si vas a imprimir o marcar el paper físicamente:

```
🟡 AMARILLO (Implementadas ✅):
   - Semantic chunking (páginas 2-3)
   - Multi-stage scoring (página 4, Algorithm líneas 11-17)
   - Dynamic thresholding (página 4, Algorithm línea 21)
   - Chunk-level filtering (páginas 1-2)

🟢 VERDE (No implementadas - Alta prioridad ❌):
   - Redundancy removal (página 2, 4, Algorithm líneas 4-10)
   - Hybrid retrieval (página 4, Algorithm líneas 2-3)

🔵 AZUL (No implementadas - Media prioridad ❌):
   - Cohere reranking (página 4, Algorithm líneas 23-24)

🔴 ROJO (Fórmulas y parámetros clave ⭐):
   - θ = 0.8 (semantic chunking threshold)
   - similarity > 0.9 (redundancy threshold)
   - T ← if var(S) < ϵ then μ + σ else μ (dynamic threshold)
   - 0.5 BM25 + 0.5 LLM (hybrid weights)
   - rerank-english-v3.0 (Cohere model)
```

---

## Checklist para Revisión del Paper

### ✅ Técnicas Implementadas

- [ ] Página 3: Semantic Chunking completo
- [ ] Página 4: Multi-stage scoring description
- [ ] Página 5, Líneas 11-17: Multi-stage scoring algorithm
- [ ] Página 5, Línea 21: Dynamic threshold formula
- [ ] Página 10: Prompts en Appendix

### ❌ Técnicas NO Implementadas

- [ ] Página 2, Sección 2.4: Redundancy description
- [ ] Página 5, Líneas 4-10: Redundancy algorithm
- [ ] Página 4: Hybrid retrieval description
- [ ] Página 5, Línea 3: Hybrid retrieval algorithm
- [ ] Página 4: Cohere reranking description
- [ ] Página 5, Línea 24: Cohere reranking algorithm

### 📊 Figuras y Tablas

- [ ] Figure 1 (página 1): Visual comparison
- [ ] Figure 2 (página 3): Complete pipeline
- [ ] Figure 3 (página 6): Redundancy effectiveness
- [ ] Table 1 (página 7): Accuracy results
- [ ] Table 2 (página 7): Threshold analysis

---

## Citas para Presentación/Defensa

### Para Explicar tu Implementación ✅

```
"Como se describe en la sección 3 del paper (página 3), implementamos
Semantic Chunking usando un threshold de θ = 0.8 para agrupar oraciones
consecutivas basándonos en similitud coseno."

"Siguiendo el Algorithm 1 (página 5, líneas 11-17), implementamos el
Multi-stage Relevance Scoring con tres etapas: LLMRelevance,
SelfReflect y CriticEval, combinadas mediante pesos 0.3-0.3-0.4."

"El Dynamic Thresholding se implementó según la línea 21 del Algorithm 1:
T ← if var(S) < ϵ then μ + σ else μ, que adapta el umbral según la
distribución de scores."
```

### Para Justificar lo No Implementado ❌

```
"El paper menciona en la sección 2.4 (página 2) y el Algorithm 1
(líneas 4-10) la técnica de Redundancy Removal con threshold > 0.9,
pero no la implementamos debido a [razón: tiempo/recursos/prioridades]."

"El Hybrid Retrieval combinando BM25 y LLM con pesos 0.5 (página 4)
no fue implementado porque [razón]."

"El Cohere Reranking (rerank-english-v3.0, página 4) requiere API
de pago y no fue priorizado en esta fase del proyecto."
```

---

## Palabras Clave del Paper (Ctrl+F)

Para buscar rápidamente en el PDF:

| Término | Apariciones | Páginas Principales |
|---------|-------------|---------------------|
| "semantic chunking" | ~10 | 2, 3, 6 |
| "multi-stage" | ~5 | 4, 5, 6 |
| "dynamic threshold" | ~8 | 4, 5, 7 |
| "redundancy" | ~12 | 2, 4, 6 |
| "BM25" | ~4 | 4, 5 |
| "Cohere" | ~3 | 4, 5 |
| "θ = 0.8" | ~3 | 3, 7 |
| "similarity > 0.9" | ~2 | 2, 4 |
| "rerank-english-v3.0" | 1 | 4 |

---

**Resumen Final**:
- **Página 5 (Algorithm 1)** es la MÁS IMPORTANTE - contiene todas las técnicas
- **Páginas 3-4** tienen las descripciones metodológicas detalladas
- **Página 10 (Appendix)** tiene los prompts exactos para implementación
