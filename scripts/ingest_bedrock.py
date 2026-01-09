"""
Ingesta a Qdrant usando Amazon Titan Embeddings V2 desde AWS Bedrock.

Pipeline 100% Bedrock (sin OpenAI):
- Chunking: Chonkie SemanticChunker con Bedrock Titan V2
- Embeddings finales: Bedrock Titan V2
- Dimensiones configurables: 256, 512, 1024
- Pricing: $0.10/1M tokens (vs $0.13/1M OpenAI + $0.02/1M OpenAI chunking)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import boto3
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import get_env as env
from pypdf import PdfReader
import logging
from typing import List

# Chonkie imports
from chonkie import SemanticChunker
from chonkie.embeddings import BaseEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
COLLECTION = "benchmark_bedrock"
MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024  # Opciones: 256, 512, 1024
REGION = "us-east-1"


class BedrockEmbeddings(BaseEmbeddings):
    """
    Bedrock Titan V2 embeddings handler compatible con Chonkie.

    Implementa BaseEmbeddings de Chonkie para usar Bedrock
    tanto en chunking como en indexing.
    """

    def __init__(self, model_id: str, dimensions: int, region: str):
        self.client = boto3.client('bedrock-runtime', region_name=region)
        self.model_id = model_id
        self._dimensions = dimensions

        # Inicializar tokenizer (usa tiktoken para compatibilidad)
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except ImportError:
            logger.warning("tiktoken not available, using simple tokenizer")
            self._tokenizer = None

        logger.info(f"Initialized Bedrock Embeddings: {model_id} ({dimensions} dims)")

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimensions

    def embed(self, text: str) -> np.ndarray:
        """
        Embed single text.

        Args:
            text: Input text

        Returns:
            Numpy array of shape (dimension,)
        """
        if not text or not text.strip():
            return np.zeros(self._dimensions, dtype=np.float32)

        request_body = {
            "inputText": text,
            "dimensions": self._dimensions,
            "normalize": True
        }

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            result = json.loads(response['body'].read())
            embedding = result['embedding']
            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed multiple texts (processed sequentially for Bedrock).

        Args:
            texts: List of input texts

        Returns:
            List of numpy arrays
        """
        return [self.embed(text) for text in texts]

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken tokenizer.

        Args:
            text: Input text

        Returns:
            Token count
        """
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        else:
            # Fallback: 1 token ≈ 4 characters
            return len(text) // 4

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts."""
        return [self.count_tokens(text) for text in texts]

    def get_tokenizer(self):
        """Get tokenizer (tiktoken cl100k_base)."""
        return self._tokenizer

    @classmethod
    def is_available(cls) -> bool:
        """Check if Bedrock embeddings are available."""
        try:
            import boto3
            return True
        except ImportError:
            return False

    def __repr__(self) -> str:
        """String representation."""
        return f"BedrockEmbeddings(model_id='{self.model_id}', dimensions={self._dimensions})"


class TitanEmbeddings:
    """
    Wrapper simple para embeddings finales (reutiliza mismo cliente Bedrock).
    Mantiene interfaz compatible con código de ingesta.
    """

    def __init__(self, bedrock_embeddings: BedrockEmbeddings):
        self.bedrock = bedrock_embeddings
        logger.info(f"TitanEmbeddings wrapper initialized")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para lista de textos.

        Args:
            texts: Lista de textos a embeddir

        Returns:
            Lista de vectores de embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        vectors = []

        for i, text in enumerate(texts, 1):
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(texts)} chunks...")

            embedding = self.bedrock.embed(text)
            vectors.append(embedding.tolist())

        logger.info(f"Generated {len(vectors)} embeddings of {self.bedrock.dimension} dimensions")
        return vectors


def load_all_pdfs_text(folder: str) -> list[dict]:
    """Carga todos los PDFs de una carpeta."""
    logger.info(f"Loading PDFs from: {folder}")
    docs = []

    for pdf_path in Path(folder).glob("*.pdf"):
        logger.info(f"  Reading: {pdf_path.name}")
        reader = PdfReader(pdf_path)
        pages_text = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

        full_text = "\n".join(pages_text).strip()
        if full_text:
            docs.append({
                "source": pdf_path.name,
                "text": full_text
            })
            logger.info(f"    Extracted {len(pages_text)} pages, {len(full_text)} chars")

    logger.info(f"Loaded {len(docs)} PDF documents\n")
    return docs


def ingest(chunks: list[str], source: str, embeddings_model: TitanEmbeddings, qdrant_client: QdrantClient):
    """
    Ingesta chunks a Qdrant usando Bedrock embeddings.

    Args:
        chunks: Lista de chunks de texto
        source: Nombre del documento fuente
        embeddings_model: Modelo de embeddings (Titan V2)
        qdrant_client: Cliente de Qdrant
    """
    logger.info(f"Uploading to Qdrant collection: {COLLECTION}")
    logger.info(f"  Chunks: {len(chunks)}")

    # Generar embeddings con Bedrock
    vectors = embeddings_model.embed_documents(chunks)

    # Upload a Qdrant
    logger.info(f"  Uploading to Qdrant...")
    qdrant_client.upload_collection(
        collection_name=COLLECTION,
        vectors=vectors,
        payload=[
            {
                "text": chunk,
                "source": source,
                "embedding_model": MODEL_ID,
                "dimensions": EMBED_DIM
            }
            for chunk in chunks
        ],
        ids=None  # Qdrant genera IDs automáticamente
    )

    logger.info(f"Uploaded {len(chunks)} chunks from {source}\n")


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("RAG INGESTION PIPELINE - 100% AWS BEDROCK")
    logger.info("=" * 80)
    logger.info(f"Embedding model: {MODEL_ID}")
    logger.info(f"Dimensions: {EMBED_DIM}")
    logger.info(f"Region: {REGION}")
    logger.info(f"Target collection: {COLLECTION}")
    logger.info(f"Chunking: Chonkie SemanticChunker + Bedrock")
    logger.info(f"Indexing: Bedrock Titan V2")
    logger.info("=" * 80 + "\n")

    # Inicializar Bedrock embeddings (usado para chunking E indexing)
    logger.info("Initializing Bedrock embeddings...")
    bedrock_embeddings = BedrockEmbeddings(
        model_id=MODEL_ID,
        dimensions=EMBED_DIM,
        region=REGION
    )

    # Wrapper para mantener compatibilidad con ingest()
    embeddings = TitanEmbeddings(bedrock_embeddings)

    # Inicializar Chonkie SemanticChunker con Bedrock
    logger.info("Initializing Chonkie SemanticChunker with Bedrock...")
    chunker = SemanticChunker(
        embedding_model=bedrock_embeddings,  # Usa Bedrock para chunking
        threshold=0.8,                        # ChunkRAG paper default
        chunk_size=128,                       # ~500 chars
        similarity_window=3,
        min_sentences_per_chunk=1,
        skip_window=0
    )
    logger.info(f"  Threshold: 0.8 (similarity)")
    logger.info(f"  Chunk size: 128 tokens (~500 chars)")
    logger.info("")

    # Inicializar Qdrant
    qdrant_client = QdrantClient(
        url=env("QDRANT_URL"),
        api_key=env("QDRANT_API_KEY"),
    )

    # Recrear colección
    logger.info(f"Recreating Qdrant collection: {COLLECTION}")
    try:
        qdrant_client.delete_collection(collection_name=COLLECTION)
        logger.info(f"  Deleted existing collection")
    except:
        logger.info(f"  Collection does not exist, creating new...")

    qdrant_client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=EMBED_DIM,
            distance=models.Distance.COSINE,
        ),
    )
    logger.info("  Collection created\n")

    # Cargar PDFs
    docs = load_all_pdfs_text("data")

    # Procesar cada documento
    for idx, doc in enumerate(docs, 1):
        text = doc["text"]
        source = doc["source"]

        logger.info("=" * 80)
        logger.info(f"Processing document {idx}/{len(docs)}: {source}")
        logger.info("=" * 80)

        # Semantic chunking con Chonkie + Bedrock
        logger.info("Semantic chunking with Chonkie + Bedrock Titan V2...")
        logger.info(f"  Input: {len(text)} chars")

        chunk_objects = chunker.chunk(text)
        chunks = [chunk.text for chunk in chunk_objects]

        avg_size = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
        logger.info(f"  Output: {len(chunks)} chunks (avg: {avg_size:.0f} chars)")

        # Ingerir a Qdrant con Bedrock embeddings
        ingest(chunks, source, embeddings, qdrant_client)

    logger.info("=" * 80)
    logger.info("INGESTION COMPLETE - 100% BEDROCK PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Processed {len(docs)} documents")
    logger.info(f"Collection: {COLLECTION}")
    logger.info(f"Embeddings: {MODEL_ID} ({EMBED_DIM} dims)")
    logger.info("")
    logger.info("Pipeline components:")
    logger.info("  1. Chunking: Chonkie + Bedrock Titan V2")
    logger.info("  2. Indexing: Bedrock Titan V2")
    logger.info("  3. No OpenAI dependencies!")
    logger.info("")
    logger.info("Cost comparison:")
    logger.info("  100% Bedrock:     ~$0.10/1M tokens (chunking + indexing)")
    logger.info("  OpenAI hybrid:    ~$0.15/1M tokens ($0.02 chunking + $0.13 indexing)")
    logger.info("  Savings:          ~33% cheaper + single vendor")
    logger.info("=" * 80)
