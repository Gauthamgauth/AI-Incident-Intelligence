"""
Advanced RAG retrieval: combines BM25 keyword search with vector
similarity search (hybrid), then reranks the merged candidates using
a cross-encoder model for higher precision.
"""

import json
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langsmith import traceable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_FILE = PROJECT_ROOT / "data" / "processed" / "incidents_clean.json"

sys.path.append(str(Path(__file__).resolve().parent))
from retriever import retrieve as vector_retrieve  # noqa: E402

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = CrossEncoder(RERANKER_MODEL)



# BM25 index (built once from all incident chunks)


def _load_corpus() -> list[dict]:
    """Builds the same chunk-level corpus used for embeddings, for BM25 indexing."""
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        incidents = json.load(f)

    corpus = []
    for inc in incidents:
        text = f"{inc['title']}. {inc.get('updates_text', '')}".strip()
        corpus.append({
            "text": text,
            "metadata": {
                "source": inc["source"],
                "incident_id": inc["id"],
                "title": inc["title"],
                "impact": inc.get("impact") or "",
                "status": inc.get("status") or "",
                "created_at": inc.get("created_at") or "",
                "shortlink": inc.get("shortlink") or "",
            },
        })
    return corpus


_corpus = _load_corpus()
_tokenized_corpus = [doc["text"].lower().split() for doc in _corpus]
_bm25 = BM25Okapi(_tokenized_corpus)


def bm25_search(query: str, n_results: int = 10) -> list[dict]:
    """Keyword-based search over the incident corpus."""
    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return [
        {"text": _corpus[i]["text"], "metadata": _corpus[i]["metadata"], "bm25_score": scores[i]}
        for i in ranked_indices
    ]


# Hybrid fusion

def hybrid_search(query: str, n_candidates: int = 10) -> list[dict]:
    """
    Merges vector search results and BM25 results into one candidate pool,
    deduped by incident_id. No scoring fusion needed here since reranking
    will re-score the merged pool anyway.
    """
    vector_results = vector_retrieve(query, n_results=n_candidates)
    keyword_results = bm25_search(query, n_results=n_candidates)

    merged = {}
    for r in vector_results + keyword_results:
        key = r["metadata"]["incident_id"]
        if key not in merged:
            merged[key] = r

    return list(merged.values())



# Reranking


def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """Re-scores candidates with a cross-encoder for higher-precision ranking."""
    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    scores = _reranker.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_n]



# Full advanced retrieval pipeline


@traceable(name="advanced_retrieve")
def advanced_retrieve(query: str, n_results: int = 5) -> list[dict]:
    candidates = hybrid_search(query, n_candidates=10)
    top_results = rerank(query, candidates, top_n=n_results)
    return top_results


if __name__ == "__main__":
    query = "what caused API outages recently"
    results = advanced_retrieve(query)

    print(f"\nQuery: {query!r}\n")
    for i, r in enumerate(results):
        meta = r["metadata"]
        print(f"--- Result {i + 1} (rerank score: {r['rerank_score']:.4f}) ---")
        print(f"Source: {meta['source']} | Impact: {meta['impact']}")
        print(f"Title: {meta['title']}")
        print(f"Text: {r['text'][:200]}...\n")