# Implementación DSPy para Optimización de Prompts

## 📚 Tabla de Contenidos
1. [¿Qué es DSPy?](#qué-es-dspy)
2. [¿Por qué DSPy en este proyecto?](#por-qué-dspy-en-este-proyecto)
3. [Arquitectura de la Implementación](#arquitectura-de-la-implementación)
4. [Proceso de Entrenamiento](#proceso-de-entrenamiento)
5. [Procesamiento Paralelo con Hilos](#procesamiento-paralelo-con-hilos)
6. [Comparación: Original vs DSPy](#comparación-original-vs-dspy)
7. [Uso Práctico](#uso-práctico)
8. [Resultados y Próximos Pasos](#resultados-y-próximos-pasos)

---

## ¿Qué es DSPy?

**DSPy** (Declarative Self-improving Language Programs, Python) es un framework de Stanford para **optimizar prompts automáticamente** usando machine learning, en lugar de escribir prompts manualmente.

### Problema que resuelve:
```python
# ❌ Enfoque tradicional: prompts hardcodeados
prompt = """
Evalúa la relevancia del siguiente chunk para la pregunta.
Chunk: {chunk}
Pregunta: {query}
Da un score de 0.0 a 1.0
"""

# Si el resultado no es bueno, tienes que:
# 1. Modificar el prompt manualmente
# 2. Probar de nuevo
# 3. Repetir hasta que funcione
```

```python
# ✅ Con DSPy: prompts optimizables
class RelevanceScore(dspy.Signature):
    """Evaluate relevance of chunk to query."""
    chunk = dspy.InputField(desc="text chunk")
    query = dspy.InputField(desc="user question")
    score = dspy.OutputField(desc="score 0.0-1.0")

scorer = dspy.ChainOfThought(RelevanceScore)

# DSPy puede optimizar este prompt AUTOMÁTICAMENTE
# usando ejemplos de entrenamiento
```

---

## ¿Por qué DSPy en este proyecto?

### Contexto del Proyecto

Este proyecto implementa **ChunkRAG** con filtrado multi-etapa:

```
Pregunta → Retrieval (15 chunks) → Multi-Stage Filtering → Top 5 chunks → Respuesta
                                          ↓
                            Etapa 1: Base Score
                            Etapa 2: Self-Reflection
                            Etapa 3: Critic Evaluation
```

### Desafío

Cada etapa usa **3 prompts diferentes** hardcodeados:
- Modificarlos manualmente es tedioso
- No sabemos si son óptimos
- Difícil de mejorar sistemáticamente

### Solución: DSPy

DSPy permite:
1. **Definir** la estructura de las 3 etapas declarativamente
2. **Entrenar** con ejemplos (chunk + query + score esperado)
3. **Optimizar** automáticamente los prompts de cada etapa
4. **Mejorar** continuamente agregando más ejemplos

---

## Arquitectura de la Implementación

### Estructura de Archivos

```
rag-engine/
├── src/filtering/
│   ├── chunk_filter.py           # ← Implementación original (asyncio)
│   └── chunk_filter_dspy.py      # ← Nueva implementación DSPy (threads)
├── pipelines/production/
│   ├── rag.py                    # ← Pipeline original
│   ├── rag_dspy.py               # ← Pipeline DSPy
│   └── compiled_scorer.json      # ← Scorer optimizado (generado)
└── scripts/
    └── train_dspy.py             # ← Script de entrenamiento
```

### Componente Principal: `MultiStageChunkScorer`

```python
class MultiStageChunkScorer(dspy.Module):
    """
    Multi-stage chunk scoring con DSPy.
    Mantiene las 3 etapas del paper ChunkRAG pero optimizables.
    """

    def __init__(self):
        super().__init__()
        # Cada etapa usa ChainOfThought para mejor razonamiento
        self.base_scorer = dspy.ChainOfThought(BaseRelevanceScore)
        self.reflector = dspy.ChainOfThought(SelfReflectionScore)
        self.critic = dspy.ChainOfThought(CriticScore)

    def forward(self, chunk: str, query: str) -> Dict[str, float]:
        # Etapa 1: Base score
        base_result = self.base_scorer(chunk=chunk, query=query)
        base_score = self._parse_score(base_result.score)

        # Etapa 2: Self-reflection
        reflect_result = self.reflector(
            chunk=chunk,
            query=query,
            base_score=str(base_score)
        )
        reflect_score = self._parse_score(reflect_result.reflected_score)

        # Etapa 3: Critic evaluation
        critic_result = self.critic(
            chunk=chunk,
            query=query,
            previous_score=str(reflect_score)
        )
        critic_score = self._parse_score(critic_result.critic_score)

        # Combinar scores (igual que en el código original)
        final_score = 0.3 * base_score + 0.3 * reflect_score + 0.4 * critic_score

        return {
            'base': base_score,
            'reflect': reflect_score,
            'critic': critic_score,
            'final': final_score
        }
```

### DSPy Signatures (Contratos de Entrada/Salida)

Las **Signatures** definen qué espera recibir y devolver cada etapa:

```python
class BaseRelevanceScore(dspy.Signature):
    """Evaluate the relevance of a text chunk to a user query."""
    chunk = dspy.InputField(desc="text chunk to evaluate")
    query = dspy.InputField(desc="user question")
    score = dspy.OutputField(desc="relevance score between 0.0 and 1.0")

class SelfReflectionScore(dspy.Signature):
    """Reflect on initial score and adjust if necessary."""
    chunk = dspy.InputField(desc="text chunk being evaluated")
    query = dspy.InputField(desc="user question")
    base_score = dspy.InputField(desc="initial relevance score")
    reflected_score = dspy.OutputField(desc="adjusted score between 0.0 and 1.0")

class CriticScore(dspy.Signature):
    """Critical evaluation: does this chunk ACTUALLY help answer the query?"""
    chunk = dspy.InputField(desc="text chunk being evaluated")
    query = dspy.InputField(desc="user question")
    previous_score = dspy.InputField(desc="score from reflection stage")
    critic_score = dspy.OutputField(desc="final critical score between 0.0 and 1.0")
```

**¿Por qué usar Signatures?**
- DSPy usa estas definiciones para **generar y optimizar prompts automáticamente**
- Los `desc` ayudan a DSPy a entender qué hace cada campo
- Son como "tipos" que guían la optimización

---

## Proceso de Entrenamiento

### 1. Crear Ejemplos de Entrenamiento

El archivo `scripts/train_dspy.py` define **8 ejemplos** con diferentes niveles de relevancia:

```python
examples = [
    # ALTA RELEVANCIA (0.8-1.0)
    dspy.Example(
        chunk="La arquitectura de Belcorp se basa en microservicios...",
        query="¿Cuál es la arquitectura de Belcorp?",
        label=0.9  # Score esperado
    ).with_inputs("chunk", "query"),

    # RELEVANCIA MEDIA (0.4-0.7)
    dspy.Example(
        chunk="La propuesta técnica incluye componentes de ML...",
        query="¿Cuál es la arquitectura de Belcorp?",
        label=0.5
    ).with_inputs("chunk", "query"),

    # BAJA RELEVANCIA (0.0-0.3)
    dspy.Example(
        chunk="El clima en Lima es templado...",
        query="¿Cuál es la arquitectura de Belcorp?",
        label=0.1
    ).with_inputs("chunk", "query"),
]
```

**Puntos clave:**
- Cada ejemplo tiene: chunk + query + label (score esperado)
- `.with_inputs()` indica qué campos son entrada (chunk, query)
- El label es la "respuesta correcta" que queremos que aprenda

### 2. Definir Métrica de Evaluación

DSPy necesita saber **qué tan bien está funcionando**:

```python
def score_accuracy_metric(example, prediction, trace=None):
    """
    Métrica: qué tan cerca está el score predicho del esperado.
    """
    predicted_score = prediction['final']
    expected_score = example.label

    # Diferencia absoluta
    diff = abs(predicted_score - expected_score)

    # Score: 1.0 si perfecto, 0.0 si muy malo
    accuracy = max(0.0, 1.0 - (diff / 0.5))

    return accuracy
```

**Ejemplo:**
- Si esperamos `0.9` y predecimos `0.85` → diff = 0.05 → accuracy = 0.90 ✅
- Si esperamos `0.9` y predecimos `0.3` → diff = 0.60 → accuracy = 0.00 ❌

### 3. Configurar el Optimizador: BootstrapFewShot

```python
optimizer = BootstrapFewShot(
    metric=score_accuracy_metric,
    max_bootstrapped_demos=4,  # Genera hasta 4 ejemplos automáticamente
    max_labeled_demos=2,        # Usa 2 ejemplos etiquetados
    max_rounds=2,               # 2 rondas de optimización
)
```

**¿Qué hace BootstrapFewShot?**

1. **Toma tus ejemplos de entrenamiento** (los 8 que definimos)
2. **Ejecuta el scorer** en cada ejemplo y ve qué tan bien funciona
3. **Selecciona los mejores ejemplos** que ayudan al modelo
4. **Agrega estos ejemplos al prompt** automáticamente (few-shot learning)
5. **Repite el proceso** por N rondas

Es como si DSPy dijera:
> "Voy a encontrar los mejores ejemplos para incluir en el prompt,
> de modo que el modelo aprenda a dar scores correctos"

### 4. Compilar (Optimizar)

```python
compiled_scorer = optimizer.compile(
    base_scorer,
    trainset=train_examples
)
```

**Durante la compilación:**
```
Bootstrapped 4 full traces after 4 examples for up to 2 rounds
```

Esto significa:
- Procesó 4 de 8 ejemplos
- Generó 4 "trazas" (ejemplos completos con razonamiento)
- Ejecutó 2 rondas de optimización

**Resultado:** Un scorer optimizado con prompts mejorados y ejemplos incluidos.

### 5. Guardar el Modelo Compilado

```python
compiled_scorer.save("pipelines/production/compiled_scorer.json")
```

Este archivo (`compiled_scorer.json`, 10 KB) contiene:
- Prompts optimizados para cada etapa
- Ejemplos few-shot seleccionados
- Configuración del scorer

**Importante:** Este archivo es lo que hace que DSPy sea mejor que hardcodear prompts.

---

## Procesamiento Paralelo con Hilos

### Desafío Inicial: DSPy no soporta AsyncIO

El código original usa `asyncio` para scoring paralelo:

```python
# ✅ Funciona con LangChain
async def score_chunk_async(chunk, query):
    response = await llm.ainvoke(prompt)  # Async nativo
    return parse_score(response)

# Ejecutar 15 chunks en paralelo
tasks = [score_chunk_async(c, q) for c in chunks]
results = await asyncio.gather(*tasks)  # 2 segundos ⚡
```

**Problema con DSPy:**
```python
# ❌ NO funciona - DSPy usa sync internamente
async def score_chunk_dspy_async(chunk, query):
    scores = scorer(chunk=chunk, query=query)  # Esto NO es async
    return scores

# Resultado: se ejecuta secuencialmente (1m49s) 🐌
```

DSPy usa llamadas síncronas a OpenAI internamente, así que `asyncio.gather()` **no las ejecuta en paralelo real**.

### Solución: ThreadPoolExecutor

Python tiene dos formas de paralelismo:
1. **AsyncIO**: Para operaciones I/O-bound asíncronas (como HTTP async)
2. **Threads**: Para operaciones síncronas que pueden ejecutarse en paralelo

Como DSPy es síncrono, usamos **threads**:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def score_chunk_dspy_sync(scorer, chunk, query, chunk_idx):
    """Función síncrona que ejecutaremos en threads."""
    logger.info(f"Chunk {chunk_idx}: Scoring with DSPy ({len(chunk)} chars)...")

    scores = scorer(chunk=chunk, query=query)  # Llamada síncrona DSPy

    logger.info(
        f"Chunk {chunk_idx}: "
        f"Base={scores['base']:.3f}, "
        f"Reflect={scores['reflect']:.3f}, "
        f"Critic={scores['critic']:.3f}, "
        f"Final={scores['final']:.3f}"
    )

    return {
        'text': chunk,
        'scores': scores,
        'final_score': scores['final']
    }

def filter_chunks_dspy_parallel(scorer, chunks, query, min_chunks=3):
    """Filtra chunks usando ThreadPoolExecutor para paralelismo."""

    scored_chunks = []

    # Crear pool de threads (máximo 15 workers)
    with ThreadPoolExecutor(max_workers=15) as executor:
        # Enviar todos los chunks al pool
        futures = {
            executor.submit(score_chunk_dspy_sync, scorer, chunk, query, i+1): i
            for i, chunk in enumerate(chunks)
        }

        # Recoger resultados a medida que terminan
        for future in as_completed(futures):
            scored_chunks.append(future.result())

    # Ordenar por orden original
    scored_chunks.sort(key=lambda x: chunks.index(x['text']))

    # Aplicar dynamic thresholding y filtrar
    # ... (resto del código)
```

### ¿Cómo funciona ThreadPoolExecutor?

**Visualmente:**

```
Main Thread
    │
    ├─→ ThreadPoolExecutor(max_workers=15)
    │       │
    │       ├─→ Thread 1: score_chunk_dspy_sync(chunk 1)  ──→ OpenAI API
    │       ├─→ Thread 2: score_chunk_dspy_sync(chunk 2)  ──→ OpenAI API
    │       ├─→ Thread 3: score_chunk_dspy_sync(chunk 3)  ──→ OpenAI API
    │       ├─→ Thread 4: score_chunk_dspy_sync(chunk 4)  ──→ OpenAI API
    │       └─→ ... (hasta 15 threads en paralelo)
    │
    ├─→ as_completed(futures)  # Espera resultados
    │       ├─ Chunk 11 terminó primero ✓
    │       ├─ Chunk 3 terminó segundo ✓
    │       ├─ Chunk 7 terminó tercero ✓
    │       └─ ... (recoge todos)
    │
    └─→ scored_chunks (todos los resultados)
```

**Log real de ejecución:**

```
16:35:18 - INFO - Chunk 1: Scoring with DSPy (189 chars)...
16:35:18 - INFO - Chunk 2: Scoring with DSPy (185 chars)...
16:35:18 - INFO - Chunk 3: Scoring with DSPy (451 chars)...
16:35:18 - INFO - Chunk 4: Scoring with DSPy (410 chars)...
...
16:35:18 - INFO - Chunk 14: Base=0.200, Reflect=0.200, Critic=0.300, Final=0.240
16:35:18 - INFO - Chunk 11: Base=0.000, Reflect=0.000, Critic=0.000, Final=0.000
16:35:18 - INFO - Chunk 10: Base=0.000, Reflect=0.000, Critic=0.000, Final=0.000
```

**Observa:** Todos empiezan a las `16:35:18` y terminan casi al mismo tiempo → **PARALELO REAL** ✅

### Comparación: Secuencial vs Paralelo

**Secuencial (MALO):**
```
Chunk 1: 16:20:57 - 16:21:06 (9s)  ──→ |████████|
Chunk 2: 16:21:06 - 16:21:13 (7s)  ──→          |███████|
Chunk 3: 16:21:13 - 16:21:21 (8s)  ──→                  |████████|
...
Total: 1m49s (109s)
```

**Paralelo con Threads (BUENO):**
```
Chunk 1:  ──→ |████████|
Chunk 2:  ──→ |███████|
Chunk 3:  ──→ |████████|
...
Total: 3s (tiempo del chunk más lento)
```

**Ganancia:** 109s → 3s = **36x más rápido** 🚀

---

## Comparación: Original vs DSPy

### Rendimiento

| Métrica | Pipeline Original | DSPy Optimizado | Diferencia |
|---------|------------------|-----------------|------------|
| Tiempo total | 2s | 3s | +1s (50% más lento) |
| Procesamiento | AsyncIO | Threads | Diferente enfoque |
| Chunks filtrados | 6 | 7 | Threshold más bajo |
| Dynamic threshold | 0.389 | 0.307 | Más permisivo |
| Prompts | Hardcoded | Optimizables | **Ventaja DSPy** |

### Scoring de Chunks (mismo query)

**Chunk 3 (Alta relevancia):**
```
Original: Base=0.800, Reflect=0.900, Critic=0.900, Final=0.870
DSPy:     Base=0.700, Reflect=0.600, Critic=0.700, Final=0.670
```

**Chunk 13 (Alta relevancia):**
```
Original: Base=0.700, Reflect=0.600, Critic=0.400, Final=0.550
DSPy:     Base=0.800, Reflect=0.700, Critic=0.800, Final=0.770
```

**Observación:** DSPy da scores **ligeramente diferentes** pero consistentes.

### Chunks Seleccionados (Top 5)

**Original:**
1. Propuesta Técnica... (0.870)
2. Arquitectura técnica... (0.830)
3. Reutilizar activos... (0.550)
4. Componentes modulares... (0.490)
5. Bloques funcionales... (0.490)

**DSPy:**
1. Propuesta Técnica... (0.770)
2. Componentes modulares... (0.670)
3. Bloques funcionales... (0.500)
4. Reutilizar activos... (0.370)
5. Conclusión... (0.370)

**Diferencia:** Ambos seleccionan chunks relevantes, pero con diferente ranking.

---

## Uso Práctico

### Opción 1: Usar Scorer Base (sin optimizar)

```bash
# Usa DSPy pero sin entrenamiento previo
uv run python pipelines/production/rag_dspy.py "¿Cuál es la arquitectura de Belcorp?"
```

**Resultado:** Usa las Signatures de DSPy con prompts por defecto.

### Opción 2: Entrenar y Usar Scorer Optimizado

```bash
# Paso 1: Entrenar (2-5 minutos, solo una vez)
uv run python scripts/train_dspy.py

# Paso 2: Usar scorer optimizado (carga automáticamente)
uv run python pipelines/production/rag_dspy.py "¿Cuál es la arquitectura de Belcorp?"
```

**Resultado:** Usa prompts optimizados + few-shot examples.

### Opción 3: Cargar Scorer Compilado Manualmente

```python
from pipelines.production import ProductionRAGDSPy
from src.filtering.chunk_filter_dspy import MultiStageChunkScorer

# Cargar scorer optimizado
scorer = MultiStageChunkScorer()
scorer.load("pipelines/production/compiled_scorer.json")

# Usar en RAG
rag = ProductionRAGDSPy(compiled_scorer=scorer)
result = rag.query("¿Qué tecnologías usa Belcorp?")
print(result['answer'])
```

### Cómo Funciona el Auto-Loading

En `rag_dspy.py`:

```python
def __init__(self, compiled_scorer=None):
    compiled_path = Path(__file__).parent / "compiled_scorer.json"

    if compiled_scorer:
        # Opción 1: Scorer pasado explícitamente
        self.scorer = compiled_scorer
    elif compiled_path.exists():
        # Opción 2: Cargar automáticamente si existe
        logger.info(f"Loading optimized scorer from {compiled_path}")
        self.scorer = MultiStageChunkScorer()
        self.scorer.load(str(compiled_path))
    else:
        # Opción 3: Usar scorer base
        logger.info("Using base DSPy scorer (not optimized)")
        self.scorer = MultiStageChunkScorer()
```

**Prioridad:**
1. Scorer pasado como parámetro
2. `compiled_scorer.json` si existe
3. Scorer base sin optimizar

---

## Resultados y Próximos Pasos

### Logros Actuales

✅ **Implementación completa de DSPy**
- 3 etapas mantenidas (base, reflect, critic)
- Procesamiento paralelo con threads (3s)
- Auto-carga de scorer optimizado
- Script de entrenamiento funcional

✅ **Optimización inicial**
- 8 ejemplos de entrenamiento
- 4 ejemplos bootstrapped
- Scorer compilado guardado (10 KB)

✅ **Rendimiento comparable**
- Original: 2s
- DSPy: 3s (+50% tiempo, pero aceptable)

### Ventajas de DSPy sobre Hardcoding

| Aspecto | Hardcoded Prompts | DSPy |
|---------|------------------|------|
| Modificación | Manual, tedioso | Automático |
| Optimización | Prueba y error | Data-driven |
| Few-shot | Escribir ejemplos manualmente | Generado automáticamente |
| Mejora continua | Difícil | Agregar ejemplos y reentrenar |
| Experimentación | Lenta | Rápida |

### Próximos Pasos Recomendados

#### 1. Evaluación con RAGAS

```bash
# Comparar métricas de ambos pipelines
python scripts/benchmark.py --pipeline original
python scripts/benchmark.py --pipeline dspy
```

**Métricas a comparar:**
- Factual Correctness
- Faithfulness
- Context Recall
- Answer Relevancy

#### 2. Ampliar Dataset de Entrenamiento

Actualmente: **8 ejemplos** → Objetivo: **50-100 ejemplos**

```python
# Agregar más ejemplos en train_dspy.py
examples = [
    # Casos edge: preguntas ambiguas
    dspy.Example(
        chunk="Azure Functions permite procesamiento serverless...",
        query="¿Cómo funciona el procesamiento?",
        label=0.6
    ),

    # Casos específicos del dominio
    dspy.Example(
        chunk="El modelo usa embeddings de 1536 dimensiones...",
        query="¿Qué dimensión tienen los embeddings?",
        label=0.95
    ),
]
```

#### 3. Experimentar con Reducción de Etapas

Actualmente: **3 etapas** (base → reflect → critic)

**Hipótesis:** Quizás 2 etapas sean suficientes.

```python
class TwoStageScorer(dspy.Module):
    def __init__(self):
        self.base_scorer = dspy.ChainOfThought(BaseRelevanceScore)
        self.critic = dspy.ChainOfThought(CriticScore)

    def forward(self, chunk, query):
        base_score = self.base_scorer(chunk=chunk, query=query)
        critic_score = self.critic(chunk=chunk, query=query, base_score=base_score)
        final_score = 0.4 * base_score + 0.6 * critic_score
        return final_score
```

**Ventajas:** Más rápido, menos llamadas API, menor costo.

#### 4. Optimizadores Avanzados

DSPy ofrece otros optimizadores:

```python
# MIPROv2: Mejor que BootstrapFewShot
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=score_accuracy_metric,
    num_candidates=10,
    init_temperature=1.0
)

# COPRO: Optimiza prompts con generación de candidatos
from dspy.teleprompt import COPRO

optimizer = COPRO(
    metric=score_accuracy_metric,
    breadth=10,
    depth=3
)
```

#### 5. A/B Testing en Producción

```python
# Usar ambos pipelines y comparar resultados
results_original = rag_original.query(question)
results_dspy = rag_dspy.query(question)

# Logging para análisis posterior
log_comparison(question, results_original, results_dspy)
```

---

## Conclusión

DSPy transforma el desarrollo de sistemas RAG de un proceso manual de "prueba y error" a uno **data-driven y sistemático**. Aunque agrega complejidad inicial, los beneficios a largo plazo (optimización automática, mejora continua, experimentación rápida) lo hacen valioso para aplicaciones de producción.

**Siguiente acción recomendada:** Ejecutar benchmarks RAGAS para validar que DSPy mantiene o mejora la calidad de respuestas.

---

**Documento actualizado:** 2026-01-07
**Versión DSPy:** 3.1.0+
**Autor:** Implementación basada en paper ChunkRAG + DSPy framework
