"""
DSPy-optimized chunk filtering with multi-stage scoring.
Mantiene las 3 etapas pero con prompts optimizables automáticamente.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from typing import List, Dict
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DSPy Signatures (definen entrada/salida de cada etapa)
# ============================================================================

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


# ============================================================================
# DSPy Module (pipeline de 3 etapas)
# ============================================================================

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
        """
        Ejecuta las 3 etapas de scoring.

        Returns:
            dict con 'base', 'reflect', 'critic', 'final'
        """
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

    def _parse_score(self, score_str: str) -> float:
        """Parse score string to float, handle errors."""
        try:
            score = float(score_str)
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            # Try to extract number from string
            import re
            numbers = re.findall(r'0?\.\d+|[01]\.?\d*', str(score_str))
            if numbers:
                return max(0.0, min(1.0, float(numbers[0])))
            logger.warning(f"Could not parse score: {score_str}, returning 0.5")
            return 0.5


# ============================================================================
# Parallel processing usando ThreadPoolExecutor
# ============================================================================

def score_chunk_dspy_sync(
    scorer: MultiStageChunkScorer,
    chunk: str,
    query: str,
    chunk_idx: int
) -> Dict[str, any]:
    """Sync scorer wrapper para paralelismo con threads."""
    logger.info(f"Chunk {chunk_idx}: Scoring with DSPy ({len(chunk)} chars)...")

    scores = scorer(chunk=chunk, query=query)

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


# ============================================================================
# Dynamic thresholding (mismo del código original)
# ============================================================================

def dynamic_threshold(scores: List[float], epsilon: float = 0.01) -> float:
    """Dynamic thresholding based on score distribution."""
    scores_array = np.array(scores)
    mean = scores_array.mean()
    std = scores_array.std()
    var = scores_array.var()

    if var < epsilon:
        threshold = mean + std
    else:
        threshold = mean

    threshold = max(0.0, min(1.0, threshold))

    logger.info(f"  Score stats: mean={mean:.3f}, std={std:.3f}, var={var:.4f}")
    logger.info(f"  Dynamic threshold: {threshold:.3f}")

    return threshold


# ============================================================================
# Main filtering function
# ============================================================================

def filter_chunks_dspy_parallel(
    scorer: MultiStageChunkScorer,
    chunks: List[str],
    query: str,
    min_chunks: int = 3
) -> List[str]:
    """
    Filter chunks usando DSPy multi-stage scorer.
    Usa ThreadPoolExecutor para paralelismo real.
    """
    if not chunks:
        return []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info(f"🚀 DSPy filtering {len(chunks)} chunks with PARALLEL scoring...")

    # Score all chunks en paralelo usando threads
    scored_chunks = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {
            executor.submit(score_chunk_dspy_sync, scorer, chunk, query, i+1): i
            for i, chunk in enumerate(chunks)
        }

        for future in as_completed(futures):
            scored_chunks.append(future.result())

    # Sort by original order
    scored_chunks.sort(key=lambda x: chunks.index(x['text']))

    # Extract final scores
    final_scores = [c['final_score'] for c in scored_chunks]

    # Dynamic threshold
    threshold = dynamic_threshold(final_scores)

    # Filter
    filtered = [
        c['text'] for c in scored_chunks
        if c['final_score'] >= threshold
    ]

    logger.info(f"Filtered from {len(chunks)} to {len(filtered)} chunks (threshold={threshold:.3f})")

    # Fallback
    if len(filtered) < min_chunks:
        logger.warning(f"Only {len(filtered)} chunks passed. Returning top {min_chunks}")
        sorted_chunks = sorted(scored_chunks, key=lambda x: x['final_score'], reverse=True)
        filtered = [c['text'] for c in sorted_chunks[:min_chunks]]

    return filtered


def filter_chunks_by_relevance_dspy(
    scorer: MultiStageChunkScorer,
    chunks: List[str],
    query: str,
    min_chunks: int = 3
) -> List[str]:
    """
    DSPy filtering with parallel processing.

    Args:
        scorer: DSPy MultiStageChunkScorer instance
        chunks: List of text chunks
        query: User query
        min_chunks: Minimum chunks to return

    Returns:
        Filtered chunks
    """
    return filter_chunks_dspy_parallel(scorer, chunks, query, min_chunks)
