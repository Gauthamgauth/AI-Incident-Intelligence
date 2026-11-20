"""
Reusable retrieval logic — embeds a query and searches ChromaDB.
Imported by search.py (testing) and the RAG generation script.
"""

import os
from pathlib import Path

from openai import OpenAI
import chromadb
from dotenv import load_dotenv
from langsmith import traceable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "incidents"
EMBEDDING_MODEL = "text-embedding-3-small"

load_dotenv(PROJECT_ROOT / ".env")
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def embed_query(query: str) -> list[float]:
    response = _client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding

@traceable(name="naive_retrieve")
def retrieve(query: str, n_results: int = 5) -> list[dict]:
    """
    Returns a list of result dicts: {text, metadata, distance}.
    Used by both the search test script and the generation pipeline.
    """
    query_embedding = embed_query(query)
    results = _collection.query(query_embeddings=[query_embedding], n_results=n_results)

    records = []
    for i in range(len(results["ids"][0])):
        records.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return records