import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings

from config import get_env as env
# from src.chunking import chonkie_chunk, semantic_chunk  # Chonkie deshabilitado
from src.chunking import semantic_chunk

from pypdf import PdfReader
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

COLLECTION = "benchmark"
EMBED_DIM = 3072  # text-embedding-3-large

client = QdrantClient(
    url=env("QDRANT_URL"),
    api_key=env("QDRANT_API_KEY"),
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=env("OPENAI_API_KEY"),
)

# ---------- PDF LOADER ----------
def load_all_pdfs_text(folder: str) -> list[dict]:
    logger.info(f"📂 Loading PDFs from: {folder}")
    docs = []

    for pdf_path in Path(folder).glob("*.pdf"):
        logger.info(f"  └─ Reading: {pdf_path.name}")
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
            logger.info(f"      ✓ Extracted {len(pages_text)} pages, {len(full_text)} chars")

    logger.info(f"✅ Loaded {len(docs)} PDF documents\n")
    return docs

# ---------- INGEST ----------
def ingest(chunks: list[str], label: str, source: str):
    collection_name = f"{COLLECTION}_{label}"

    logger.info(f"📤 Uploading to Qdrant collection: {collection_name}")
    logger.info(f"  └─ Generating embeddings for {len(chunks)} chunks...")
    vectors = embeddings.embed_documents(chunks)
    logger.info(f"  └─ Embeddings generated: {len(vectors)} vectors of {EMBED_DIM} dimensions")

    logger.info(f"  └─ Uploading to Qdrant...")
    client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        payload=[
            {
                "text": c,
                "chunker": label,
                "source": source
            }
            for c in chunks
        ],
        ids=None,  # Qdrant genera IDs automáticamente
    )

    logger.info(f"✅ Uploaded {len(chunks)} chunks from {source} to {collection_name}\n")

# ---------- MAIN ----------
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🚀 RAG INGESTION PIPELINE - STARTING")
    logger.info("=" * 80)
    logger.info(f"Target collection: benchmark_semantic (ChunkRAG)")
    logger.info(f"Embedding model: text-embedding-3-large ({EMBED_DIM} dimensions)")
    logger.info("=" * 80 + "\n")

    # Recrea colección semantic
    logger.info("🗑️  Recreating Qdrant collection...")
    # SOLO SEMANTIC CHUNKING (ChunkRAG paper)
    collection_name = f"{COLLECTION}_semantic"
    logger.info(f"  └─ Recreating: {collection_name}")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=EMBED_DIM,
            distance=models.Distance.COSINE,
        ),
    )

    # CHONKIE DESHABILITADO - Solo se usa semantic chunking para producción
    # for label in ["chonkie", "semantic"]:
    #     collection_name = f"{COLLECTION}_{label}"
    #     logger.info(f"  └─ Recreating: {collection_name}")
    #     client.recreate_collection(
    #         collection_name=collection_name,
    #         vectors_config=models.VectorParams(
    #             size=EMBED_DIM,
    #             distance=models.Distance.COSINE,
    #         ),
    #     )

    logger.info("✅ Collection recreated\n")

    docs = load_all_pdfs_text("data")

    for idx, doc in enumerate(docs, 1):
        text = doc["text"]
        source = doc["source"]

        logger.info("=" * 80)
        logger.info(f"📄 Processing document {idx}/{len(docs)}: {source}")
        logger.info("=" * 80 + "\n")

        # CHONKIE DESHABILITADO - Solo se usa semantic chunking
        # logger.info("--- CHONKIE (Token-based) ---")
        # chunks_chonkie = chonkie_chunk(text)
        # ingest(chunks_chonkie, "chonkie", source)

        # Semantic chunking (ChunkRAG - Producción)
        logger.info("--- SEMANTIC CHUNKING (ChunkRAG) ---")
        chunks_semantic = semantic_chunk(text)
        ingest(chunks_semantic, "semantic", source)

    logger.info("=" * 80)
    logger.info("🎉 INGESTION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Processed {len(docs)} documents")
    logger.info("Collection ready:")
    logger.info("  ✅ benchmark_semantic (ChunkRAG semantic chunks, θ=0.8)")
    logger.info("")
    logger.info("Chonkie chunking disabled (production uses semantic only)")
    logger.info("=" * 80)
