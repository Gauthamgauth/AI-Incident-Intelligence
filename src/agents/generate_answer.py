"""
Full RAG pipeline: retrieves relevant incidents, then asks GPT-4 mini
to synthesize a grounded natural-language answer citing sources.
"""

import os
import sys
from pathlib import Path

from openai import OpenAI
from langsmith import traceable 
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "retrieval"))  # so we can import retriever.py

from retriever import retrieve  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")
print("LangSmith key loaded:", bool(os.getenv("LANGCHAIN_API_KEY")))
print("LangSmith tracing enabled:", os.getenv("LANGCHAIN_TRACING_V2"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o-mini"


def build_context(results: list[dict]) -> str:
    """Formats retrieved chunks into a context block for the prompt."""
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


@traceable(name="naive_rag_generate")
def generate_answer(query: str, n_results: int = 5) -> str:
    results = retrieve(query, n_results)
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
        temperature=0.2,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    query = "what caused API outages recently"
    answer = generate_answer(query)
    print(f"\nQuery: {query}\n")
    print(f"Answer:\n{answer}")