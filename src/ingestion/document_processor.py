"""
Generic document processor with automatic metadata generation.

Features:
- Document-agnostic processing (PDFs, text, etc.)
- Semantic chunking with Chonkie + Cohere Embed v4
- RAPTOR hierarchical clustering (automatic cluster metadata)
- BM25 keyword extraction (automatic keyword metadata)
- Optional: Anthropic Contextual Retrieval

No manual metadata field definition - fully automatic.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from pypdf import PdfReader

from chonkie import SemanticChunker

from ..embeddings import CohereChonkieEmbeddings, CohereEmbedV4
from ..clustering import RAPTORClustering, ClusterMetadata
from ..indexing import BM25KeywordExtractor, KeywordMetadata

logger = logging.getLogger(__name__)


@dataclass
class ProcessedChunk:
    """
    Chunk with automatic metadata.

    All metadata is generated automatically without manual field definition.
    """
    # Core content
    text: str
    source: str
    chunk_index: int

    # Embeddings
    embedding: List[float]

    # RAPTOR clustering metadata (automatic)
    cluster_id: str
    tree_level: int
    cluster_size: int
    centroid_similarity: float
    parent_cluster_id: Optional[str] = None

    # BM25 keyword metadata (automatic)
    keywords: List[str] = None
    keyword_scores: List[float] = None
    top_keyword: Optional[str] = None
    keyword_diversity: float = 0.0

    # Optional: Contextual metadata
    context_prefix: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for Qdrant payload."""
        data = asdict(self)
        # Convert numpy array to list if needed
        if isinstance(data['embedding'], np.ndarray):
            data['embedding'] = data['embedding'].tolist()
        return data


@dataclass
class DocumentMetadata:
    """Document-level metadata."""
    source: str
    n_chunks: int
    total_chars: int
    avg_chunk_size: float
    n_clusters: int
    global_keywords: List[str]
    global_keyword_scores: List[float]


class DocumentProcessor:
    """
    Generic document processor with automatic metadata generation.

    Uses RAPTOR + BM25 for automatic hierarchical and keyword metadata.
    No manual field definition required.
    """

    def __init__(
        self,
        # Embeddings config
        model_id: str = "cohere.embed-v4:0",
        region: str = "us-east-1",
        dimensions: int = 1024,
        # Chunking config
        chunk_size: int = 128,
        threshold: float = 0.8,
        similarity_window: int = 3,
        min_sentences: int = 1,
        # Clustering config
        n_levels: int = 2,
        reduction_factor: int = 10,
        # Keyword config
        top_k_keywords: int = 5,
        ngram_range: tuple = (1, 2),
        # Optional features
        enable_contextual_retrieval: bool = False
    ):
        """
        Initialize document processor.

        Args:
            model_id: Bedrock model ID for Cohere Embed v4
            region: AWS region
            dimensions: Embedding dimensions
            chunk_size: Chunk size in tokens
            threshold: Semantic similarity threshold for chunking
            similarity_window: Window for similarity calculation
            min_sentences: Minimum sentences per chunk
            n_levels: Number of RAPTOR hierarchy levels
            reduction_factor: Cluster reduction per level
            top_k_keywords: Keywords to extract per chunk
            ngram_range: N-gram range for keywords
            enable_contextual_retrieval: Use Anthropic's Contextual Retrieval
        """
        # Initialize Cohere embeddings for chunking
        logger.info("Initializing Cohere Embed v4 for chunking...")
        self.chunking_embeddings = CohereChonkieEmbeddings(
            model_id=model_id,
            region=region,
            dimensions=dimensions
        )

        # Initialize Cohere embeddings for final indexing
        logger.info("Initializing Cohere Embed v4 for indexing...")
        self.indexing_embeddings = CohereEmbedV4(
            model_id=model_id,
            region=region,
            dimensions=dimensions
        )

        # Initialize semantic chunker
        logger.info("Initializing Chonkie SemanticChunker...")
        self.chunker = SemanticChunker(
            embedding_model=self.chunking_embeddings,
            threshold=threshold,
            chunk_size=chunk_size,
            similarity_window=similarity_window,
            min_sentences_per_chunk=min_sentences,
            skip_window=0
        )

        # Initialize RAPTOR clustering
        logger.info("Initializing RAPTOR clustering...")
        self.clusterer = RAPTORClustering(
            n_levels=n_levels,
            reduction_factor=reduction_factor
        )

        # Initialize BM25 keyword extractor
        logger.info("Initializing BM25 keyword extractor...")
        self.keyword_extractor = BM25KeywordExtractor(
            top_k=top_k_keywords,
            ngram_range=ngram_range
        )

        self.enable_contextual_retrieval = enable_contextual_retrieval
        self.dimensions = dimensions

        logger.info(
            "DocumentProcessor initialized",
            extra={
                "model_id": model_id,
                "dimensions": dimensions,
                "chunk_size": chunk_size,
                "threshold": threshold,
                "n_levels": n_levels,
                "contextual_retrieval": enable_contextual_retrieval
            }
        )

    def load_pdf(self, file_path: Path) -> str:
        """
        Load text from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text
        """
        logger.info(f"Loading PDF: {file_path.name}")

        reader = PdfReader(file_path)
        pages_text = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

        full_text = "\n".join(pages_text).strip()

        logger.info(
            f"PDF loaded: {file_path.name}",
            extra={
                "n_pages": len(pages_text),
                "n_chars": len(full_text)
            }
        )

        return full_text

    def load_text(self, file_path: Path) -> str:
        """
        Load text from text file.

        Args:
            file_path: Path to text file

        Returns:
            File contents
        """
        logger.info(f"Loading text file: {file_path.name}")

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        logger.info(
            f"Text file loaded: {file_path.name}",
            extra={"n_chars": len(text)}
        )

        return text

    def load_document(self, file_path: Path) -> str:
        """
        Load document from file (auto-detects type).

        Args:
            file_path: Path to document

        Returns:
            Extracted text
        """
        suffix = file_path.suffix.lower()

        if suffix == '.pdf':
            return self.load_pdf(file_path)
        elif suffix in ['.txt', '.md', '.text']:
            return self.load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def chunk_document(self, text: str) -> List[str]:
        """
        Chunk document using semantic chunking.

        Args:
            text: Document text

        Returns:
            List of chunk texts
        """
        logger.info("Chunking document with Chonkie SemanticChunker...")

        chunk_objects = self.chunker.chunk(text)
        chunks = [chunk.text for chunk in chunk_objects]

        avg_size = sum(len(c) for c in chunks) / len(chunks) if chunks else 0

        logger.info(
            "Chunking complete",
            extra={
                "n_chunks": len(chunks),
                "avg_chunk_size": avg_size
            }
        )

        return chunks

    def generate_embeddings(self, chunks: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of chunk texts

        Returns:
            List of embeddings
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")

        embeddings = self.indexing_embeddings.embed_batch(
            chunks,
            input_type="search_document"
        )

        logger.info(
            "Embeddings generated",
            extra={
                "n_embeddings": len(embeddings),
                "dimensions": self.dimensions
            }
        )

        return embeddings

    def apply_clustering(
        self,
        embeddings: List[np.ndarray]
    ) -> List[ClusterMetadata]:
        """
        Apply RAPTOR hierarchical clustering.

        Args:
            embeddings: List of chunk embeddings

        Returns:
            List of cluster metadata (level 0)
        """
        logger.info("Applying RAPTOR hierarchical clustering...")

        all_levels = self.clusterer.cluster_hierarchical(embeddings)

        if not all_levels:
            logger.warning("No clustering results")
            return []

        # Return level 0 (original chunks) metadata
        # NOTE: Levels 1+ contain centroid embeddings without associated text.
        # Full RAPTOR requires LLM-generated summaries for higher levels.
        # For now, we only index level 0 with hierarchical cluster metadata.
        # Future: Implement LLM summarization for levels 1-2 to enable tree-based retrieval.
        level_0_metadata = all_levels[0]

        logger.info(
            "Clustering complete",
            extra={
                "n_levels": len(all_levels),
                "n_clusters_l0": len(set(m.cluster_id for m in level_0_metadata))
            }
        )

        return level_0_metadata

    def generate_context_prefix(
        self,
        source: str,
        global_keywords: List[str],
        text: str
    ) -> str:
        """
        Generate contextual prefix for chunks (Anthropic Contextual Retrieval).

        Creates document-level context to prepend to each chunk, improving
        retrieval accuracy by providing global document context.

        Args:
            source: Document source filename
            global_keywords: Top keywords from the document
            text: Full document text

        Returns:
            Context prefix string
        """
        if not self.enable_contextual_retrieval:
            return None

        # Extract document type from filename
        doc_type = source.rsplit('.', 1)[0].replace('_', ' ')

        # Get top 5 keywords
        top_keywords = global_keywords[:5] if global_keywords else []

        # Calculate basic stats
        word_count = len(text.split())
        char_count = len(text)

        # Build context prefix
        # Format inspired by Anthropic's Contextual Retrieval
        context_parts = [f"Document: {doc_type}"]

        if top_keywords:
            keywords_str = ", ".join(top_keywords)
            context_parts.append(f"Main topics: {keywords_str}")

        context_parts.append(f"Length: {word_count} words")

        context_prefix = " | ".join(context_parts)

        return context_prefix

    def apply_keywords(self, chunks: List[str]) -> List[KeywordMetadata]:
        """
        Apply BM25 keyword extraction.

        Args:
            chunks: List of chunk texts

        Returns:
            List of keyword metadata
        """
        logger.info("Applying BM25 keyword extraction...")

        keyword_metadata = self.keyword_extractor.extract_keywords(chunks)

        logger.info(
            "Keyword extraction complete",
            extra={
                "n_chunks": len(keyword_metadata),
                "avg_keywords": np.mean([len(m.keywords) for m in keyword_metadata])
            }
        )

        return keyword_metadata

    def process_document(
        self,
        file_path: Path
    ) -> tuple[List[ProcessedChunk], DocumentMetadata]:
        """
        Process single document with automatic metadata generation.

        Args:
            file_path: Path to document file

        Returns:
            Tuple of (processed chunks, document metadata)
        """
        logger.info("=" * 80)
        logger.info(f"Processing document: {file_path.name}")
        logger.info("=" * 80)

        # 1. Load document
        text = self.load_document(file_path)

        # 2. Chunk document
        chunks = self.chunk_document(text)

        if not chunks:
            logger.warning(f"No chunks generated for {file_path.name}")
            return [], None

        # 3. Generate embeddings
        embeddings = self.generate_embeddings(chunks)

        # 4. Apply RAPTOR clustering
        cluster_metadata = self.apply_clustering(embeddings)

        # 5. Apply BM25 keywords
        keyword_metadata = self.apply_keywords(chunks)

        # 6. Extract global keywords
        global_keywords, global_scores = self.keyword_extractor.extract_global_keywords(
            chunks,
            top_k=20
        )

        # 7. Generate context prefix (Anthropic Contextual Retrieval)
        context_prefix = self.generate_context_prefix(
            source=file_path.name,
            global_keywords=global_keywords,
            text=text
        )

        if context_prefix:
            logger.info(
                "Generated contextual prefix",
                extra={"prefix": context_prefix[:100]}
            )

        # 8. Combine all metadata
        processed_chunks = []

        for idx, (chunk_text, embedding, cluster_meta, keyword_meta) in enumerate(
            zip(chunks, embeddings, cluster_metadata, keyword_metadata)
        ):
            processed_chunk = ProcessedChunk(
                # Core content
                text=chunk_text,
                source=file_path.name,
                chunk_index=idx,
                # Embeddings
                embedding=embedding.tolist(),
                # RAPTOR metadata
                cluster_id=cluster_meta.cluster_id,
                tree_level=cluster_meta.tree_level,
                cluster_size=cluster_meta.cluster_size,
                centroid_similarity=cluster_meta.centroid_similarity,
                parent_cluster_id=cluster_meta.parent_cluster_id,
                # BM25 metadata
                keywords=keyword_meta.keywords,
                keyword_scores=keyword_meta.keyword_scores,
                top_keyword=keyword_meta.top_keyword,
                keyword_diversity=keyword_meta.keyword_diversity,
                # Contextual retrieval
                context_prefix=context_prefix
            )

            processed_chunks.append(processed_chunk)

        # 9. Create document metadata
        avg_chunk_size = sum(len(c) for c in chunks) / len(chunks)
        n_clusters = len(set(m.cluster_id for m in cluster_metadata))

        doc_metadata = DocumentMetadata(
            source=file_path.name,
            n_chunks=len(chunks),
            total_chars=len(text),
            avg_chunk_size=avg_chunk_size,
            n_clusters=n_clusters,
            global_keywords=global_keywords,
            global_keyword_scores=global_scores
        )

        logger.info(
            "Document processing complete",
            extra={
                "source": file_path.name,
                "n_chunks": len(processed_chunks),
                "n_clusters": n_clusters,
                "top_keyword": global_keywords[0] if global_keywords else None
            }
        )

        return processed_chunks, doc_metadata

    def process_directory(
        self,
        directory: Path,
        pattern: str = "*.pdf"
    ) -> tuple[List[ProcessedChunk], List[DocumentMetadata]]:
        """
        Process all documents in directory.

        Args:
            directory: Directory containing documents
            pattern: File pattern (e.g., "*.pdf", "*.txt")

        Returns:
            Tuple of (all chunks, all document metadata)
        """
        logger.info("=" * 80)
        logger.info(f"Processing directory: {directory}")
        logger.info(f"Pattern: {pattern}")
        logger.info("=" * 80)

        all_chunks = []
        all_metadata = []

        file_paths = sorted(directory.glob(pattern))

        if not file_paths:
            logger.warning(f"No files matching pattern '{pattern}' in {directory}")
            return [], []

        logger.info(f"Found {len(file_paths)} files\n")

        for idx, file_path in enumerate(file_paths, 1):
            logger.info(f"\n[{idx}/{len(file_paths)}] Processing: {file_path.name}")

            try:
                chunks, metadata = self.process_document(file_path)

                all_chunks.extend(chunks)
                all_metadata.append(metadata)

                logger.info(f"✓ Successfully processed {file_path.name}\n")

            except Exception as e:
                logger.error(
                    f"✗ Failed to process {file_path.name}",
                    extra={"error": str(e)},
                    exc_info=True
                )
                continue

        logger.info("=" * 80)
        logger.info("DIRECTORY PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Files processed: {len(all_metadata)}/{len(file_paths)}")
        logger.info(f"Total chunks: {len(all_chunks)}")
        logger.info(f"Avg chunks per document: {len(all_chunks) / len(all_metadata) if all_metadata else 0:.1f}")
        logger.info("=" * 80)

        return all_chunks, all_metadata
