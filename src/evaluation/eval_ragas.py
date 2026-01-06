import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset
from langchain_openai import ChatOpenAI
import logging
import json
from datetime import datetime

from src.retrieval import retrieve
from src.evaluation.ground_truth import GROUND_TRUTH

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# LLM para generar respuestas y para RAGAS
llm_chat = ChatOpenAI(model="gpt-4o-mini")
evaluator_llm = LangchainLLMWrapper(llm_chat)

QUESTIONS = [
    # ACTIVE: 5 preguntas representativas (~15-20 min de evaluación)

    # 1. Pregunta sobre arquitectura general (Belcorp)
    "¿Cuál es el objetivo principal de la arquitectura propuesta para Belcorp?",

    # 2. Pregunta sobre flujo completo (Learning Journey)
    "¿Cuál es el flujo completo del sistema Learning Journey desde que se envía el correo hasta que se recibe la respuesta?",

    # 3. Pregunta técnica específica (fórmula de ranking)
    "¿Cuál es la fórmula de ranking utilizada para puntuar los cursos en Learning Journey y qué factores considera?",

    # 4. Pregunta sobre componentes técnicos
    "¿Qué componentes de Azure se utilizan en el proyecto Learning Journey BCP y cuál es la función de cada uno?",

    # 5. Pregunta sobre embeddings (técnica detallada)
    "¿Qué modelo de embeddings se utiliza en Learning Journey y cuántas dimensiones tiene?"

    # COMMENTED OUT: Para evaluación rápida (descomenta para evaluación completa)
    # "¿Qué componentes de ML menciona la propuesta técnica de Belcorp y para qué sirven?",
    # "¿Por qué la propuesta de Belcorp no está alineada a un stack tecnológico específico?",
    # "¿Qué beneficios se esperan de la plataforma de analítica avanzada de Belcorp y cómo se logran?",
    # "¿Cómo funciona la búsqueda híbrida en el sistema Learning Journey y qué fuentes de datos utiliza?",
    # "¿Qué validaciones se aplican al archivo Excel de entrada en el sistema Learning Journey?",
    # "¿Cómo se realiza la deduplicación de cursos en el sistema Learning Journey?",
    # "¿Qué similitudes hay entre los enfoques de arquitectura de ambos documentos en términos de componentes de datos?",
    # "¿Qué medidas de seguridad se mencionan en el proyecto Learning Journey BCP?",
    # "¿Qué algoritmo se utiliza para la selección de cursos por capacidad en Learning Journey?",
    # "¿Cuáles son los KPIs técnicos clave que se monitorean en el sistema Learning Journey?",
]

def save_results_to_json(results, label, use_filtering, dataset_rows):
    """
    Save evaluation results to JSON file with detailed metrics.

    Args:
        results: RAGAS evaluation results (EvaluationResult object)
        label: Chunker label (chonkie/llama)
        use_filtering: Whether filtering was enabled
        dataset_rows: Original dataset rows for context
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "filtered" if use_filtering else "baseline"

    # Create results directory structure
    results_dir = Path(__file__).parent.parent.parent / "results" / mode
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = results_dir / f"results_{mode}_{label}_{timestamp}.json"

    # Extract metrics from results - use to_pandas() to get scores
    try:
        results_df = results.to_pandas()
        # Get mean scores for each metric
        metrics_dict = {}
        for col in results_df.columns:
            if col not in ['user_input', 'retrieved_contexts', 'response', 'reference']:
                mean_value = results_df[col].mean()
                metrics_dict[col] = float(mean_value) if str(mean_value) != 'nan' else None
    except Exception as e:
        logging.warning(f"Could not extract metrics from results: {e}")
        # Fallback: try to access scores dict directly
        metrics_dict = {}
        if hasattr(results, 'scores'):
            for metric_name, score in results.scores.items():
                metrics_dict[metric_name] = float(score) if str(score) != 'nan' else None

    # Build detailed JSON structure
    output = {
        "metadata": {
            "timestamp": timestamp,
            "chunker": label,
            "mode": "ChunkRAG Filtering" if use_filtering else "Baseline (No Filtering)",
            "filtering_enabled": use_filtering,
            "num_questions": len(QUESTIONS),
            "embedding_model": "text-embedding-3-large",
            "llm_model": "gpt-4o-mini"
        },
        "metrics_summary": {
            "context_recall": {
                "value": metrics_dict.get('context_recall'),
                "description": "Proporción de información necesaria que fue recuperada (mayor es mejor)",
                "interpretation": "Mide si se recuperó toda la información necesaria para responder"
            },
            "faithfulness": {
                "value": metrics_dict.get('faithfulness'),
                "description": "Qué tan fiel es la respuesta al contexto recuperado (mayor es mejor)",
                "interpretation": "Mide si el LLM alucinó o se mantuvo fiel al contexto"
            },
            "factual_correctness": {
                "value": metrics_dict.get('factual_correctness(mode=f1)', metrics_dict.get('factual_correctness')),
                "description": "Qué tan correcta es la respuesta comparada con la referencia (mayor es mejor)",
                "interpretation": "Mide la precisión factual de la respuesta (F1 score)"
            }
        },
        "configuration": {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "retrieval_k": 5,
            "retrieval_initial_k": 15 if use_filtering else 5,
            "filtering_stages": ["base_score", "self_reflection", "critic_eval"] if use_filtering else None,
            "dynamic_threshold_enabled": use_filtering
        },
        "questions_evaluated": [
            {
                "index": i + 1,
                "question": q,
                "answer_length_chars": len(dataset_rows[i]['response']),
                "num_contexts_retrieved": len(dataset_rows[i]['retrieved_contexts'])
            }
            for i, q in enumerate(QUESTIONS)
        ],
        "raw_results": metrics_dict
    }

    # Save to JSON
    output_path = Path(filename)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logging.info(f"\nResults saved to: {filename}")
    return filename


def build_dataset(label, use_filtering=False):
    """
    Build dataset with optional ChunkRAG filtering.

    Args:
        label: Chunker label (chonkie/llama)
        use_filtering: If True, apply LLM-based chunk filtering
    """
    filter_suffix = " WITH FILTERING" if use_filtering else ""
    logging.info(f"Building dataset for chunker: {label}{filter_suffix}")
    rows = []
    for i, q in enumerate(QUESTIONS, 1):
        logging.info(f"  Question {i}/{len(QUESTIONS)}: {q[:80]}...")

        logging.info(f"    Retrieving contexts from Qdrant...")
        contexts = retrieve(q, label, use_filtering=use_filtering)
        logging.info(f"    Retrieved {len(contexts)} chunks")

        logging.info(f"    Generating answer with LLM...")
        answer = llm_chat.invoke(
            f"Responde usando solo este contexto:\n{contexts}"
        ).content
        logging.info(f"    Answer generated ({len(answer)} chars)")

        # RAGAS format: user_input, retrieved_contexts, response, reference
        rows.append({
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": GROUND_TRUTH.get(q, "")  # Use ground truth answer
        })

    logging.info(f"Dataset built: {len(rows)} examples\n")
    return EvaluationDataset.from_list(rows), rows

if __name__ == "__main__":
    import sys

    # Check if filtering mode is requested
    use_filtering = "--filter" in sys.argv or "-f" in sys.argv

    if use_filtering:
        logging.info("=" * 60)
        logging.info("ChunkRAG MODE: LLM-based Chunk Filtering ENABLED")
        logging.info("=" * 60)
    else:
        logging.info("=" * 60)
        logging.info("BASELINE MODE: No filtering (vanilla retrieval)")
        logging.info("To enable filtering, run: uv run eval_ragas.py --filter")
        logging.info("=" * 60)

    logging.info(f"\nTotal questions: {len(QUESTIONS)}")
    logging.info(f"Estimated time: ~{len(QUESTIONS) * 2 * 2} minutes for both chunkers")
    logging.info(f"Chunkers to evaluate: chonkie, semantic\n")

    saved_files = []

    for label in ["chonkie", "semantic"]:
        logging.info(f"\n{'='*60}")
        mode = "WITH LLM FILTERING" if use_filtering else "BASELINE (No Filtering)"
        logging.info(f"Evaluating: {label.upper()} - {mode}")
        logging.info(f"{'='*60}\n")

        ds, rows = build_dataset(label, use_filtering=use_filtering)

        logging.info(f"Computing RAGAS metrics for {label}...")
        result = evaluate(
            dataset=ds,
            metrics=[
                LLMContextRecall(),
                Faithfulness(),
                FactualCorrectness(),
            ],
            llm=evaluator_llm,
        )

        print(f"\n{'='*60}")
        mode_str = " + ChunkRAG FILTERING" if use_filtering else " (BASELINE)"
        print(f"RESULTS - {label.upper()}{mode_str}")
        print(f"{'='*60}")
        print(result)
        print()

        # Save results to JSON
        filename = save_results_to_json(result, label, use_filtering, rows)
        saved_files.append(filename)

    logging.info("\n" + "="*60)
    logging.info("EVALUATION COMPLETED")
    logging.info("="*60)
    logging.info(f"Results saved to:")
    for file in saved_files:
        logging.info(f"  - {file}")
    logging.info("\nTo compare baseline vs filtered, run both commands:")
    logging.info("  uv run eval_ragas.py           # Baseline")
    logging.info("  uv run eval_ragas.py --filter  # With ChunkRAG filtering")
    logging.info("="*60)
