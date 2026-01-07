"""
Script para optimizar los prompts de DSPy con ejemplos de entrenamiento.

Uso:
    uv run python scripts/train_dspy.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from dspy.teleprompt import BootstrapFewShot
from config import get_env as env
from src.filtering.chunk_filter_dspy import MultiStageChunkScorer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Ejemplos de entrenamiento
# ============================================================================

def create_training_examples():
    """
    Crea ejemplos de entrenamiento para optimizar DSPy.

    Cada ejemplo tiene:
    - chunk: texto del chunk
    - query: pregunta del usuario
    - label: score esperado (0.0-1.0)
    """
    examples = [
        # ALTA RELEVANCIA (0.8-1.0)
        dspy.Example(
            chunk="La arquitectura de Belcorp se basa en microservicios con Azure como plataforma cloud principal.",
            query="¿Cuál es la arquitectura de Belcorp?",
            label=0.9
        ).with_inputs("chunk", "query"),

        dspy.Example(
            chunk="El sistema Learning Journey utiliza embeddings de 1536 dimensiones generados con text-embedding-ada-002.",
            query="¿Qué modelo de embeddings se utiliza en Learning Journey?",
            label=0.95
        ).with_inputs("chunk", "query"),

        # RELEVANCIA MEDIA (0.4-0.7)
        dspy.Example(
            chunk="La propuesta técnica incluye componentes de ML para análisis predictivo y clasificación.",
            query="¿Cuál es la arquitectura de Belcorp?",
            label=0.5
        ).with_inputs("chunk", "query"),

        dspy.Example(
            chunk="El proyecto utiliza Azure Cosmos DB como base de datos principal para almacenamiento.",
            query="¿Qué componentes de Azure se utilizan?",
            label=0.6
        ).with_inputs("chunk", "query"),

        # BAJA RELEVANCIA (0.0-0.3)
        dspy.Example(
            chunk="El clima en Lima es templado durante todo el año, con temperaturas promedio de 20°C.",
            query="¿Cuál es la arquitectura de Belcorp?",
            label=0.1
        ).with_inputs("chunk", "query"),

        dspy.Example(
            chunk="La historia de la empresa se remonta a 1968 cuando fue fundada en Perú.",
            query="¿Qué tecnologías usa el sistema?",
            label=0.2
        ).with_inputs("chunk", "query"),

        # MÁS EJEMPLOS REALISTAS
        dspy.Example(
            chunk="Los componentes principales son API Gateway, ML Services, y Azure Functions para procesamiento serverless.",
            query="¿Qué componentes tiene la arquitectura?",
            label=0.9
        ).with_inputs("chunk", "query"),

        dspy.Example(
            chunk="El flujo inicia cuando el usuario envía un email que es procesado por Azure Logic Apps.",
            query="¿Cuál es el flujo del sistema Learning Journey?",
            label=0.85
        ).with_inputs("chunk", "query"),
    ]

    logger.info(f"Created {len(examples)} training examples")
    return examples


# ============================================================================
# Métrica de evaluación
# ============================================================================

def score_accuracy_metric(example, prediction, trace=None):
    """
    Métrica: qué tan cerca está el score predicho del esperado.

    Returns:
        float: 1.0 si perfecto, 0.0 si muy malo
    """
    try:
        # El prediction es el resultado de scorer.forward()
        predicted_score = prediction['final']
        expected_score = example.label

        # Diferencia absoluta
        diff = abs(predicted_score - expected_score)

        # Score: 1.0 si diff=0, 0.0 si diff>=0.5
        accuracy = max(0.0, 1.0 - (diff / 0.5))

        return accuracy
    except Exception as e:
        logger.error(f"Metric error: {e}")
        return 0.0


# ============================================================================
# Optimización
# ============================================================================

def optimize_scorer():
    """
    Optimiza el MultiStageChunkScorer con BootstrapFewShot.

    Returns:
        Optimized scorer
    """
    logger.info("="*60)
    logger.info("OPTIMIZING DSPy MULTI-STAGE SCORER")
    logger.info("="*60)

    # Configurar DSPy
    dspy.configure(lm=dspy.LM('openai/gpt-4o-mini', api_key=env("OPENAI_API_KEY")))

    # Crear ejemplos
    train_examples = create_training_examples()

    # Crear scorer base
    base_scorer = MultiStageChunkScorer()

    # Optimizador
    logger.info("\nConfiguring BootstrapFewShot optimizer...")
    optimizer = BootstrapFewShot(
        metric=score_accuracy_metric,
        max_bootstrapped_demos=4,  # Agrega hasta 4 ejemplos automáticamente
        max_labeled_demos=2,        # Usa 2 ejemplos etiquetados
        max_rounds=2,               # 2 rondas de optimización
    )

    # Compilar (optimizar)
    logger.info("\nCompiling scorer (this may take a few minutes)...")
    logger.info("Running optimization rounds...")

    compiled_scorer = optimizer.compile(
        base_scorer,
        trainset=train_examples
    )

    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION COMPLETE!")
    logger.info("="*60)

    return compiled_scorer


# ============================================================================
# Guardar modelo compilado
# ============================================================================

def save_compiled_scorer(scorer, filename="compiled_scorer.json"):
    """Guarda el scorer compilado para uso futuro."""
    save_path = Path(__file__).parent.parent / "pipelines" / "production" / filename

    logger.info(f"\nSaving compiled scorer to: {save_path}")
    scorer.save(str(save_path))
    logger.info("✅ Scorer saved successfully!")

    return save_path


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DSPy PROMPT OPTIMIZATION")
    print("="*60)
    print("\nThis script will:")
    print("1. Create training examples")
    print("2. Optimize prompts using BootstrapFewShot")
    print("3. Save the compiled scorer")
    print("\nEstimated time: 2-5 minutes")
    print("="*60 + "\n")

    # Optimizar
    compiled_scorer = optimize_scorer()

    # Guardar
    save_path = save_compiled_scorer(compiled_scorer)

    print("\n" + "="*60)
    print("HOW TO USE THE OPTIMIZED SCORER:")
    print("="*60)
    print(f"""
# Load the compiled scorer:
from src.filtering.chunk_filter_dspy import MultiStageChunkScorer
scorer = MultiStageChunkScorer()
scorer.load("{save_path}")

# Or use with ProductionRAGDSPy:
from pipelines.production.rag_dspy import ProductionRAGDSPy
rag = ProductionRAGDSPy(compiled_scorer=scorer)
""")
    print("="*60)
