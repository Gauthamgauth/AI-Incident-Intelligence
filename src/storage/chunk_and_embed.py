"""
Chunks each incident's text, generates embeddings via OpenAI's
text-embedding-3-small, and stores everything in a persistent ChromaDB
collection for semantic retrieval.
"""

import os
import json
import logging
from pathlib import Path

from openai import OpenAI
import chromadb
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_FILE = PROJECT_ROOT / "data" / "processed" / "incidents_clean.json"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

load_dotenv(PROJECT_ROOT / ".env")

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800       
CHUNK_OVERLAP = 100     
COLLECTION_NAME = "incidents"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# Chunking

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window character chunker. Good enough for incident narratives."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_chunks(incident: dict) -> list[dict]:
    """
    Builds one or more chunk records for an incident. Each chunk carries
    the title (for context) plus a slice of updates_text, and inherits
    the incident's metadata for filtering later.
    """
    full_text = f"{incident['title']}. {incident.get('updates_text', '')}".strip()
    text_chunks = chunk_text(full_text)

    records = []
    for i, chunk in enumerate(text_chunks):
        records.append({
            "chunk_id": f"{incident['source']}_{incident['id']}_chunk{i}",
            "text": chunk,
            "metadata": {
                "source": incident["source"],
                "incident_id": incident["id"],
                "title": incident["title"],
                "impact": incident.get("impact") or "",
                "status": incident.get("status") or "",
                "created_at": incident.get("created_at") or "",
                "shortlink": incident.get("shortlink") or "",
            },
        })
    return records



# Embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Calls OpenAI embeddings API in a single batch request."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]



# Orchestration


def run() -> None:
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        incidents = json.load(f)
    logger.info(f"Loaded {len(incidents)} incidents")

    all_chunks = []
    for inc in incidents:
        all_chunks.extend(build_chunks(inc))
    logger.info(f"Built {len(all_chunks)} chunks from {len(incidents)} incidents")

    # Embed in batches (OpenAI allows large batches; 100 is a safe, well-tested size)
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)
        all_embeddings.extend(embeddings)
        logger.info(f"Embedded batch {i // batch_size + 1} ({len(batch)} chunks)")

    # Store in ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    collection.upsert(
        ids=[c["chunk_id"] for c in all_chunks],
        embeddings=all_embeddings,
        documents=[c["text"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
    )

    logger.info(f"Stored {len(all_chunks)} chunks in ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
    print(f"\nDone. Total chunks embedded and stored: {len(all_chunks)}")


if __name__ == "__main__":
    run()