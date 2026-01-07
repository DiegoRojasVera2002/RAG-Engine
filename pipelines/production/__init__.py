from .rag import ProductionRAG
try:
    from .rag_dspy import ProductionRAGDSPy
    __all__ = ["ProductionRAG", "ProductionRAGDSPy"]
except ImportError:
    # DSPy no instalado
    __all__ = ["ProductionRAG"]
