"""
Production RAG pipeline using DSPy + Enhanced Retrieval V2.

Integrates:
- Enhanced metadata retrieval (query_v2_enhanced.py)
- DSPy prompting with ChainOfThought
- Flexible LLM backends (OpenAI, Anthropic, Bedrock)
- Full pipeline: retrieve → filter → rerank → generate
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from typing import List, Optional, Literal
import logging
from dataclasses import dataclass

from src.retrieval.query_v2_enhanced import (
    retrieve_with_metadata,
    retrieve_simple,
    retrieve_with_filtering,
    retrieve_with_reranking,
    retrieve_full_pipeline
)
from config import get_env as env

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================================
# DSPy Signatures (prompts estructurados)
# ============================================================================

class GenerateAnswer(dspy.Signature):
    """Answer questions based on provided context chunks."""
    
    question = dspy.InputField(desc="User's question")
    context = dspy.InputField(desc="Retrieved context chunks")
    answer = dspy.OutputField(desc="Comprehensive answer based on context")


class GenerateAnswerWithReasoning(dspy.Signature):
    """
    Answer questions with detailed, well-structured responses.
    All responses must be in Spanish.
    
    Requirements:
    - Provide specific information from the context with supporting details
    - Explain WHY each piece of information is relevant to the question
    - Use structured format (numbered lists or bullet points) for clarity
    - Include comparative analysis when multiple items are found
    - Add contextual details that enhance understanding
    - Be thorough and avoid generic statements
    """
    
    question = dspy.InputField(desc="User's question")
    context = dspy.InputField(desc="Retrieved information chunks with metadata")
    
    reasoning = dspy.OutputField(
        desc="""Detailed step-by-step analysis (in Spanish) explaining:
        - Which information from the context answers the question
        - Why each piece of information is relevant
        - Key details that support the conclusions
        - How different pieces of information relate or compare
        """
    )
    
    answer = dspy.OutputField(
        desc="""Comprehensive, structured answer (in Spanish) that includes:
        1. Specific information extracted from the context
        2. Relevant details and attributes found in the source material
        3. Clear explanations of WHY each item is relevant to the question
        4. Contextual information that adds clarity and depth
        5. Comparative insights when multiple items match the criteria
        6. Professional formatting (bullet points, numbered lists, or well-organized paragraphs)
        
        The answer should be thorough, specific, and explain the reasoning behind conclusions.
        Avoid vague or generic statements - provide concrete details from the context.
        """
    )


# ============================================================================
# DSPy Modules (componentes reutilizables)
# ============================================================================

class RAGModule(dspy.Module):
    """
    Basic RAG module: Retrieve → Generate.
    
    Uses simple vector search + direct answer generation.
    """
    
    def __init__(self, k: int = 5):
        super().__init__()
        self.k = k
        self.generate = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question: str):
        """Execute RAG pipeline."""
        # Retrieve
        logging.info(f"Retrieving {self.k} chunks for: {question[:50]}...")
        chunks = retrieve_simple(question, k=self.k)
        
        # Format context
        context = self._format_context(chunks)
        
        # Generate answer
        logging.info("Generating answer with DSPy...")
        result = self.generate(question=question, context=context)
        
        return dspy.Prediction(
            answer=result.answer,
            context=context,
            num_chunks=len(chunks)
        )
    
    def _format_context(self, chunks: List[str]) -> str:
        """Format chunks into context string."""
        return "\n\n".join([
            f"[{i+1}] {chunk}"
            for i, chunk in enumerate(chunks)
        ])


class RAGModuleEnhanced(dspy.Module):
    """
    Enhanced RAG module with metadata and reasoning.
    
    Pipeline:
    1. Retrieve with metadata (keywords, context, clusters)
    2. Optional filtering/reranking
    3. Generate answer with reasoning
    """
    
    def __init__(
        self,
        k: int = 5,
        use_filtering: bool = False,
        use_reranking: bool = False,
        use_reasoning: bool = True
    ):
        super().__init__()
        self.k = k
        self.use_filtering = use_filtering
        self.use_reranking = use_reranking
        self.use_reasoning = use_reasoning
        
        # Choose signature based on reasoning flag
        signature = GenerateAnswerWithReasoning if use_reasoning else GenerateAnswer
        self.generate = dspy.ChainOfThought(signature)
    
    def forward(self, question: str):
        """Execute enhanced RAG pipeline."""
        # Retrieve with full metadata
        logging.info(f"Enhanced retrieval (k={self.k}, filter={self.use_filtering}, rerank={self.use_reranking})...")
        chunks_meta = retrieve_with_metadata(
            question,
            k=self.k,
            use_filtering=self.use_filtering,
            use_reranking=self.use_reranking
        )
        
        # Format context with metadata
        context = self._format_context_with_metadata(chunks_meta)
        
        # Generate answer
        logging.info("Generating answer with DSPy...")
        result = self.generate(question=question, context=context)
        
        # Build prediction
        prediction = dspy.Prediction(
            answer=result.answer,
            context=context,
            num_chunks=len(chunks_meta),
            metadata=chunks_meta
        )
        
        # Add reasoning if available
        if self.use_reasoning and hasattr(result, 'reasoning'):
            prediction.reasoning = result.reasoning
        
        return prediction
    
    def _format_context_with_metadata(self, chunks_meta: List[dict]) -> str:
        """Format chunks with enhanced metadata."""
        formatted = []
        
        for i, chunk in enumerate(chunks_meta, 1):
            # Start with chunk number
            lines = [f"[{i}]"]
            
            # Add context prefix if available (Anthropic Contextual Retrieval)
            if chunk.get("context_prefix"):
                lines.append(f"Context: {chunk['context_prefix']}")
            
            # Add main content
            lines.append(f"Content: {chunk['text']}")
            
            # Add metadata footer
            meta_parts = []
            if chunk.get("top_keyword"):
                meta_parts.append(f"Topic: {chunk['top_keyword']}")
            if chunk.get("source"):
                meta_parts.append(f"Source: {chunk['source']}")
            if chunk.get("score"):
                meta_parts.append(f"Score: {chunk['score']:.3f}")
            
            if meta_parts:
                lines.append(f"({', '.join(meta_parts)})")
            
            formatted.append("\n".join(lines))
        
        return "\n\n".join(formatted)


# ============================================================================
# Pipeline Factory (configuración fácil)
# ============================================================================

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    
    # Retrieval
    k: int = 5
    use_filtering: bool = False
    use_reranking: bool = False
    
    # Generation
    use_reasoning: bool = True
    
    # LLM backend
    llm_provider: Literal["openai", "anthropic", "bedrock"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2000


def create_rag_pipeline(config: RAGConfig = None) -> RAGModuleEnhanced:
    """
    Factory function to create configured RAG pipeline.
    
    Args:
        config: RAGConfig instance (uses defaults if None)
    
    Returns:
        Configured RAGModuleEnhanced instance
    
    Example:
        >>> config = RAGConfig(k=5, use_reranking=True)
        >>> rag = create_rag_pipeline(config)
        >>> result = rag("¿Quién tiene experiencia con AWS?")
        >>> print(result.answer)
    """
    if config is None:
        config = RAGConfig()
    
    # Configure LLM
    logging.info(f"Configuring DSPy with {config.llm_provider} ({config.model})...")
    
    if config.llm_provider == "openai":
        lm = dspy.LM(
            model=f"openai/{config.model}",
            api_key=env("OPENAI_API_KEY"),
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    elif config.llm_provider == "anthropic":
        lm = dspy.LM(
            model=f"anthropic/{config.model}",
            api_key=env("ANTHROPIC_API_KEY"),
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    elif config.llm_provider == "bedrock":
        # Bedrock support via LiteLLM
        lm = dspy.LM(
            model=f"bedrock/{config.model}",
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
    
    dspy.configure(lm=lm)
    
    # Create pipeline
    pipeline = RAGModuleEnhanced(
        k=config.k,
        use_filtering=config.use_filtering,
        use_reranking=config.use_reranking,
        use_reasoning=config.use_reasoning
    )
    
    logging.info(f"RAG pipeline created: k={config.k}, filter={config.use_filtering}, rerank={config.use_reranking}")
    
    return pipeline


# ============================================================================
# Convenience Functions
# ============================================================================

def query_rag(
    question: str,
    k: int = 5,
    use_filtering: bool = False,
    use_reranking: bool = False,
    model: str = "gpt-4o-mini"
) -> dict:
    """
    One-liner RAG query function.
    
    Args:
        question: User question
        k: Number of chunks to retrieve
        use_filtering: Apply LLM filtering
        use_reranking: Apply Cohere reranking
        model: LLM model to use
    
    Returns:
        Dict with keys: answer, reasoning (optional), context, metadata
    
    Example:
        >>> result = query_rag("¿Quién sabe Python?", k=3, use_reranking=True)
        >>> print(result["answer"])
    """
    config = RAGConfig(
        k=k,
        use_filtering=use_filtering,
        use_reranking=use_reranking,
        model=model
    )
    
    pipeline = create_rag_pipeline(config)
    prediction = pipeline(question)
    
    return {
        "answer": prediction.answer,
        "reasoning": getattr(prediction, "reasoning", None),
        "context": prediction.context,
        "num_chunks": prediction.num_chunks,
        "metadata": prediction.metadata
    }


# ============================================================================
# Main (Testing)
# ============================================================================

if __name__ == "__main__":
    """Test the production RAG pipeline."""
    
    print("\n" + "="*80)
    print("PRODUCTION RAG PIPELINE (DSPy + Enhanced Retrieval V2)")
    print("="*80 + "\n")
    
    # Test queries
    test_queries = [
        "¿Quién tiene experiencia con AWS y cloud computing?",
        "¿Qué tecnologías de IA mencionan los candidatos?",
        "¿Quién tiene experiencia en desarrollo backend?"
    ]
    
    query = test_queries[2]
    
    # ============================================================================
    # TEST 1: Baseline (Comentado - descomenta para comparar)
    # ============================================================================
    # print("\n" + "="*80)
    # print("QUERY: " + query)
    # print("="*80 + "\n")
    # 
    # print("\n--- Baseline ---")
    # print("Config: k=3, filter=False, rerank=False")
    # config_baseline = RAGConfig(k=3, use_filtering=False, use_reranking=False)
    # try:
    #     pipeline_baseline = create_rag_pipeline(config_baseline)
    #     result_baseline = pipeline_baseline(query)
    #     
    #     print(f"\nAnswer:\n{result_baseline.answer}")
    #     if hasattr(result_baseline, 'reasoning'):
    #         print(f"\nReasoning:\n{result_baseline.reasoning}")
    #     print(f"\nMetadata: {result_baseline.num_chunks} chunks retrieved")
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     logging.exception("Baseline failed")
    
    # ============================================================================
    # TEST 2: With Filtering (Comentado - descomenta para comparar)
    # ============================================================================
    # print("\n--- With Filtering ---")
    # print("Config: k=5, filter=True, rerank=False")
    # config_filtering = RAGConfig(k=5, use_filtering=True, use_reranking=False)
    # try:
    #     pipeline_filtering = create_rag_pipeline(config_filtering)
    #     result_filtering = pipeline_filtering(query)
    #     
    #     print(f"\nAnswer:\n{result_filtering.answer}")
    #     if hasattr(result_filtering, 'reasoning'):
    #         print(f"\nReasoning:\n{result_filtering.reasoning}")
    #     print(f"\nMetadata: {result_filtering.num_chunks} chunks retrieved")
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     logging.exception("Filtering pipeline failed")
    
    # ============================================================================
    # TEST 3: Full Pipeline (ACTIVO - el mejor)
    # ============================================================================
    print("\n" + "="*80)
    print("QUERY: " + query)
    print("="*80 + "\n")
    
    print("--- Full Pipeline ---")
    print("Config: k=5, filter=True, rerank=True")
    config = RAGConfig(
        k=5, 
        use_filtering=True, 
        use_reranking=True, 
        model="gpt-4o-mini"
    )
    
    try:
        pipeline = create_rag_pipeline(config)
        result = pipeline(query)
        
        print(f"\n{'='*80}")
        print("ANSWER")
        print(f"{'='*80}")
        print(result.answer)
        
        if hasattr(result, 'reasoning'):
            print(f"\n{'='*80}")
            print("REASONING")
            print(f"{'='*80}")
            print(result.reasoning)
        
        print(f"\n{'='*80}")
        print("METADATA")
        print(f"{'='*80}")
        print(f"Chunks retrieved: {result.num_chunks}")
        
        # Opcional: Mostrar top 3 chunks sources
        if hasattr(result, 'metadata') and result.metadata:
            print("\nTop sources:")
            for i, chunk in enumerate(result.metadata[:3], 1):
                source = chunk.get('source', 'Unknown')
                score = chunk.get('score', 0.0)
                print(f"  {i}. {source} (score: {score:.3f})")
        
    except Exception as e:
        print(f"ERROR: {e}")
        logging.exception("Pipeline failed")
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}\n")
