"""
Quick semantic search test against the ChromaDB incidents collection.
Takes a natural-language query, embeds it, and returns the most
similar incident chunks.
"""

import os
from pathlib import Path

from openai import OpenAI
import chromadb
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "incidents"
EMBEDDING_MODEL = "text-embedding-3-small"

load_dotenv(PROJECT_ROOT / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_query(query: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


def search(query: str, n_results: int = 5) -> None:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    print(f"\nQuery: {query!r}\n")
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        distance = results["distances"][0][i]

        print(f"--- Result {i + 1} (distance: {distance:.4f}) ---")
        print(f"Source: {meta['source']} | Impact: {meta['impact']} | Status: {meta['status']}")
        print(f"Title: {meta['title']}")
        print(f"Text: {doc[:200]}...")
        print(f"Link: {meta['shortlink']}\n")


if __name__ == "__main__":
    
    search("what caused API outages recently")


"""
Quick CLI test for retrieval — prints results for a given query.
"""

from retriever import retrieve


def print_results(query: str, n_results: int = 5) -> None:
    results = retrieve(query, n_results)

    print(f"\nQuery: {query!r}\n")
    for i, r in enumerate(results):
        meta = r["metadata"]
        print(f"--- Result {i + 1} (distance: {r['distance']:.4f}) ---")
        print(f"Source: {meta['source']} | Impact: {meta['impact']} | Status: {meta['status']}")
        print(f"Title: {meta['title']}")
        print(f"Text: {r['text'][:200]}...")
        print(f"Link: {meta['shortlink']}\n")


if __name__ == "__main__":
    print_results("what caused API outages recently")