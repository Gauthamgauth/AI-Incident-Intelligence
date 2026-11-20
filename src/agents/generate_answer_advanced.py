"""
Advanced RAG generation: uses hybrid search + reranking for retrieval,
then GPT-4o-mini for grounded answer generation.
"""

import os
import sys
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from langsmith import traceable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "retrieval"))

from hybrid_retriever import advanced_retrieve

load_dotenv(PROJECT_ROOT / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o-mini"


def build_context(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results):
        meta = r["metadata"]
        blocks.append(
            f"[Source {i + 1}] ({meta['source']}, impact: {meta['impact']}, status: {meta['status']})\n"
            f"Title: {meta['title']}\n"
            f"Details: {r['text']}\n"
            f"Link: {meta['shortlink']}"
        )
    return "\n\n".join(blocks)


@traceable(name="advanced_rag_generate")
def generate_answer(query: str, n_results: int = 5) -> str:
    results = advanced_retrieve(query, n_results)
    context = build_context(results)

    system_prompt = (
        "You are an incident intelligence assistant. Answer the user's question "
        "using ONLY the incident data provided below. Cite sources using [Source N] "
        "notation. If the context doesn't contain enough information to answer, say so "
        "clearly instead of guessing."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    query = "what happened in github recently on nov 13"
    answer = generate_answer(query)
    print(f"\nQuery: {query}\n")
    print(f"Answer:\n{answer}")