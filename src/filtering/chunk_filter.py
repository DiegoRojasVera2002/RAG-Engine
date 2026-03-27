"""
Chunk-level LLM filtering based on ChunkRAG paper (arXiv:2410.19572v5)
Multi-stage relevance scoring: Base → Self-Reflection → Critic
Optimized with async parallel processing
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from config import get_env as env
import numpy as np
from typing import List, Dict
import logging
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=env("OPENAI_API_KEY"),
    temperature=0.0  # Deterministic for scoring
)


def llm_relevance_score(chunk: str, query: str) -> float:
    """
    Base LLM relevance scoring.
    Returns a score between 0 and 1.
    """
    prompt = f"""You are an AI assistant tasked with determining the relevance of a text chunk to a user query.

Analyze the provided chunk and query, then assign a relevance score between 0 and 1, where 1 means highly relevant and 0 means not relevant at all.

Chunk: {chunk}

User Query: {query}

Provide ONLY a single decimal number between 0 and 1, representing the relevance score. No other text.

Relevance Score (between 0 and 1):"""

    try:
        response = llm.invoke(prompt).content.strip()
        score = float(response)
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    except Exception as e:
        logging.warning(f"LLM scoring failed: {e}. Returning 0.5")
        return 0.5


def self_reflect_score(chunk: str, query: str, base_score: float) -> float:
    """
    Self-reflection: LLM reflects on its own scoring and adjusts if necessary.
    """
    prompt = f"""You have assigned a relevance score to a text chunk based on a user query.

Your initial score was: {base_score}

Reflect on your scoring and adjust the score if necessary. Provide the final score.

Chunk: {chunk}

User Query: {query}

Provide ONLY a single decimal number between 0 and 1, representing the final relevance score. No other text.

Final Relevance Score (between 0 and 1):"""

    try:
        response = llm.invoke(prompt).content.strip()
        score = float(response)
        return max(0.0, min(1.0, score))
    except Exception as e:
        logging.warning(f"Self-reflection failed: {e}. Returning base score")
        return base_score


def critic_eval(chunk: str, query: str, reflected_score: float) -> float:
    """
    Critic evaluation: Apply domain-specific heuristics.
    For now, we'll do a simple verification pass.
    """
    prompt = f"""You are a critical evaluator reviewing a relevance score assigned to a text chunk.

The previous score was: {reflected_score}

Apply critical thinking and domain-specific verification. Does this chunk ACTUALLY help answer the query? Be strict.

Chunk: {chunk}

User Query: {query}

Provide ONLY a single decimal number between 0 and 1, representing your critical evaluation score. No other text.

Critical Score (between 0 and 1):"""

    try:
        response = llm.invoke(prompt).content.strip()
        score = float(response)
        return max(0.0, min(1.0, score))
    except Exception as e:
        logging.warning(f"Critic eval failed: {e}. Returning reflected score")
        return reflected_score


def combine_scores(base: float, reflect: float, critic: float) -> float:
    """
    Combine multi-stage scores.
    Using weighted average: base (0.3) + reflect (0.3) + critic (0.4)
    Critic gets highest weight as it's the most refined.
    """
    return 0.3 * base + 0.3 * reflect + 0.4 * critic


def score_chunk_relevance(chunk: str, query: str) -> Dict[str, float]:
    """
    Multi-stage chunk scoring following ChunkRAG methodology.

    Returns:
        dict with 'base', 'reflect', 'critic', and 'final' scores
    """
    logging.info(f"  Scoring chunk ({len(chunk)} chars)...")

    # Stage 1: Base LLM scoring
    base = llm_relevance_score(chunk, query)
    logging.info(f"    Base score: {base:.3f}")

    # Stage 2: Self-reflection
    reflect = self_reflect_score(chunk, query, base)
    logging.info(f"    Reflected score: {reflect:.3f}")

    # Stage 3: Critic evaluation
    critic = critic_eval(chunk, query, reflect)
    logging.info(f"    Critic score: {critic:.3f}")

    # Combine
    final = combine_scores(base, reflect, critic)
    logging.info(f"    Final combined score: {final:.3f}")

    return {
        'base': base,
        'reflect': reflect,
        'critic': critic,
        'final': final
    }


def dynamic_threshold(scores: List[float], epsilon: float = 0.01) -> float:
    """
    Dynamic thresholding based on score distribution (ChunkRAG Algorithm 1, line 21).

    If variance is low (scores are tight), use μ + σ to be more selective.
    Otherwise, use just μ.
    """
    scores_array = np.array(scores)
    mean = scores_array.mean()
    std = scores_array.std()
    var = scores_array.var()

    if var < epsilon:
        threshold = mean + std
    else:
        threshold = mean

    # Clamp threshold to [0, 1]
    threshold = max(0.0, min(1.0, threshold))

    logging.info(f"  Score stats: mean={mean:.3f}, std={std:.3f}, var={var:.4f}")
    logging.info(f"  Dynamic threshold: {threshold:.3f}")

    return threshold


async def llm_relevance_score_async(chunk: str, query: str) -> float:
    """Async version of base LLM relevance scoring."""
    prompt = f"""You are an AI assistant tasked with determining the relevance of a text chunk to a user query.

Analyze the provided chunk and query, then assign a relevance score between 0 and 1, where 1 means highly relevant and 0 means not relevant at all.

Chunk: {chunk}

User Query: {query}

Provide ONLY a single decimal number between 0 and 1, representing the relevance score. No other text.

Relevance Score (between 0 and 1):"""

    try:
        response = await llm.ainvoke(prompt)
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except Exception as e:
        logging.warning(f"LLM scoring failed: {e}. Returning 0.5")
        return 0.5


async def self_reflect_score_async(chunk: str, query: str, base_score: float) -> float:
    """Async version of self-reflection scoring."""
    prompt = f"""You have assigned a relevance score to a text chunk based on a user query.

Your initial score was: {base_score}

Reflect on your scoring and adjust the score if necessary. Provide the final score.

Chunk: {chunk}

User Query: {query}

Provide ONLY a single decimal number between 0 and 1, representing the final relevance score. No other text.

Final Relevance Score (between 0 and 1):"""

    try:
        response = await llm.ainvoke(prompt)
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except Exception as e:
        logging.warning(f"Self-reflection failed: {e}. Returning base score")
        return base_score


async def critic_eval_async(chunk: str, query: str, reflected_score: float) -> float:
    """Async version of critic evaluation."""
    prompt = f"""You are a critical evaluator reviewing a relevance score assigned to a text chunk.

The previous score was: {reflected_score}

Apply critical thinking and domain-specific verification. Does this chunk ACTUALLY help answer the query? Be strict.

Chunk: {chunk}

User Query: {query}

Provide ONLY a single decimal number between 0 and 1, representing your critical evaluation score. No other text.

Critical Score (between 0 and 1):"""

    try:
        response = await llm.ainvoke(prompt)
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except Exception as e:
        logging.warning(f"Critic eval failed: {e}. Returning reflected score")
        return reflected_score


async def score_chunk_relevance_async(chunk: str, query: str, chunk_idx: int) -> Dict[str, any]:
    """
    Async version of multi-stage chunk scoring.
    Allows parallel processing of multiple chunks.
    """
    logging.info(f"Chunk {chunk_idx}: Scoring ({len(chunk)} chars)...")

    # Run all 3 stages in sequence (but this function runs in parallel with others)
    base = await llm_relevance_score_async(chunk, query)
    logging.info(f"Chunk {chunk_idx}: Base={base:.3f}")

    reflect = await self_reflect_score_async(chunk, query, base)
    logging.info(f"Chunk {chunk_idx}: Reflect={reflect:.3f}")

    critic = await critic_eval_async(chunk, query, reflect)
    logging.info(f"Chunk {chunk_idx}: Critic={critic:.3f}")

    final = combine_scores(base, reflect, critic)
    logging.info(f"Chunk {chunk_idx}: Final={final:.3f}")

    return {
        'text': chunk,
        'scores': {
            'base': base,
            'reflect': reflect,
            'critic': critic,
            'final': final
        },
        'final_score': final
    }


async def filter_chunks_async(chunks: List[str], query: str, min_chunks: int = 3) -> List[str]:
    """
    Async parallel version of chunk filtering.
    Processes all chunks in parallel using asyncio.gather.

    ~3x faster than sequential version.
    """
    if not chunks:
        return []

    logging.info(f"🚀 Filtering {len(chunks)} chunks with PARALLEL async scoring...")

    # Score all chunks in parallel
    tasks = [
        score_chunk_relevance_async(chunk, query, i+1)
        for i, chunk in enumerate(chunks)
    ]
    scored_chunks = await asyncio.gather(*tasks)

    # Extract final scores for thresholding
    final_scores = [c['final_score'] for c in scored_chunks]

    # Determine dynamic threshold
    threshold = dynamic_threshold(final_scores)

    # Filter chunks above threshold
    filtered = [
        c['text'] for c in scored_chunks
        if c['final_score'] >= threshold
    ]

    logging.info(f"Filtered from {len(chunks)} to {len(filtered)} chunks (threshold={threshold:.3f})")

    # Fallback: if too few chunks, return top min_chunks
    if len(filtered) < min_chunks:
        logging.warning(f"Only {len(filtered)} chunks passed. Returning top {min_chunks}")
        sorted_chunks = sorted(scored_chunks, key=lambda x: x['final_score'], reverse=True)
        filtered = [c['text'] for c in sorted_chunks[:min_chunks]]

    return filtered


def filter_chunks_by_relevance(
    chunks: List[str],
    query: str,
    min_chunks: int = 3,
    use_async: bool = True
) -> List[str]:
    """
    Filter chunks using multi-stage LLM scoring and dynamic thresholding.

    Args:
        chunks: List of text chunks to filter
        query: User query
        min_chunks: Minimum number of chunks to return (fallback)
        use_async: If True, use parallel async processing (3x faster)

    Returns:
        Filtered list of relevant chunks
    """
    if not chunks:
        return []

    if use_async:
        # Use async parallel processing
        return asyncio.run(filter_chunks_async(chunks, query, min_chunks))

    # Fallback to sequential processing
    logging.info(f"Filtering {len(chunks)} chunks with LLM scoring (sequential)...")

    # Score all chunks
    scored_chunks = []
    for i, chunk in enumerate(chunks, 1):
        logging.info(f"Chunk {i}/{len(chunks)}:")
        scores = score_chunk_relevance(chunk, query)
        scored_chunks.append({
            'text': chunk,
            'scores': scores,
            'final_score': scores['final']
        })

    # Extract final scores for thresholding
    final_scores = [c['final_score'] for c in scored_chunks]

    # Determine dynamic threshold
    threshold = dynamic_threshold(final_scores)

    # Filter chunks above threshold
    filtered = [
        c['text'] for c in scored_chunks
        if c['final_score'] >= threshold
    ]

    logging.info(f"Filtered from {len(chunks)} to {len(filtered)} chunks (threshold={threshold:.3f})")

    # Fallback: if too few chunks, return top min_chunks
    if len(filtered) < min_chunks:
        logging.warning(f"Only {len(filtered)} chunks passed. Returning top {min_chunks}")
        sorted_chunks = sorted(scored_chunks, key=lambda x: x['final_score'], reverse=True)
        filtered = [c['text'] for c in sorted_chunks[:min_chunks]]

    return filtered
