"""
Evaluation harness for the RAG pipeline using LLM-as-judge.
Scores retrieval relevance, answer faithfulness, and answer relevance
for a fixed set of test queries. Saves results to data/eval_results/
for comparing naive vs advanced RAG later.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "retrieval"))
sys.path.append(str(PROJECT_ROOT / "src" / "agents"))

from retriever import retrieve  # noqa: E402
from generate_answer import generate_answer, build_context  # noqa: E402
from hybrid_retriever import advanced_retrieve  # noqa: E402
import generate_answer_advanced  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(PROJECT_ROOT / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_MODEL = "gpt-4o-mini"
TEST_QUERIES_FILE = Path(__file__).resolve().parent / "test_queries.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval_results"



# LLM-as-judge scoring functions


def _judge(prompt: str) -> dict:
    """Calls the judge model and expects strict JSON back: {score, reason}."""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict evaluator. Respond with ONLY valid JSON, no markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def score_retrieval_relevance(query: str, context: str) -> dict:
    prompt = f"""Rate how relevant the retrieved context is to the query, on a scale of 1-5.
5 = all retrieved chunks are directly relevant. 1 = none are relevant.

Query: {query}

Retrieved context:
{context}

Respond with JSON: {{"score": <1-5 integer>, "reason": "<one sentence>"}}"""
    return _judge(prompt)


def score_faithfulness(answer: str, context: str) -> dict:
    prompt = f"""Rate whether the answer is fully supported by the given context, on a scale of 1-5.
5 = every claim in the answer is backed by the context (no hallucination).
1 = the answer contains claims not found in the context at all.

Context:
{context}

Answer:
{answer}

Respond with JSON: {{"score": <1-5 integer>, "reason": "<one sentence>"}}"""
    return _judge(prompt)


def score_answer_relevance(query: str, answer: str) -> dict:
    prompt = f"""Rate how well the answer addresses the query, on a scale of 1-5.
5 = directly and completely answers the question. 1 = does not address it at all.

Query: {query}

Answer:
{answer}

Respond with JSON: {{"score": <1-5 integer>, "reason": "<one sentence>"}}"""
    return _judge(prompt)



# Orchestration


def evaluate_query(query: str, pipeline: str = "naive", n_results: int = 5) -> dict:
    if pipeline == "naive":
        results = retrieve(query, n_results)
        context = build_context(results)
        answer = generate_answer(query, n_results)
    else:  # "advanced"
        results = advanced_retrieve(query, n_results)
        context = generate_answer_advanced.build_context(results)
        answer = generate_answer_advanced.generate_answer(query, n_results)

    retrieval_score = score_retrieval_relevance(query, context)
    faithfulness_score = score_faithfulness(answer, context)
    relevance_score = score_answer_relevance(query, answer)

    return {
        "query": query,
        "answer": answer,
        "retrieval_relevance": retrieval_score,
        "faithfulness": faithfulness_score,
        "answer_relevance": relevance_score,
    }


def run(pipeline_name: str = "naive_rag", pipeline: str = "naive") -> None:
    with open(TEST_QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)

    logger.info(f"Running evaluation on {len(queries)} queries for pipeline: {pipeline_name}")

    all_results = []
    for i, query in enumerate(queries):
        logger.info(f"[{i + 1}/{len(queries)}] Evaluating: {query}")
        result = evaluate_query(query, pipeline=pipeline)
        all_results.append(result)

    # Aggregate averages
    avg_retrieval = sum(r["retrieval_relevance"]["score"] for r in all_results) / len(all_results)
    avg_faithfulness = sum(r["faithfulness"]["score"] for r in all_results) / len(all_results)
    avg_relevance = sum(r["answer_relevance"]["score"] for r in all_results) / len(all_results)

    summary = {
        "pipeline": pipeline_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_queries": len(all_results),
        "avg_retrieval_relevance": round(avg_retrieval, 2),
        "avg_faithfulness": round(avg_faithfulness, 2),
        "avg_answer_relevance": round(avg_relevance, 2),
        "details": all_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{pipeline_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n--- Evaluation Summary: {pipeline_name} ---")
    print(f"Avg Retrieval Relevance : {summary['avg_retrieval_relevance']} / 5")
    print(f"Avg Faithfulness        : {summary['avg_faithfulness']} / 5")
    print(f"Avg Answer Relevance    : {summary['avg_answer_relevance']} / 5")
    print(f"\nSaved full results -> {out_path}")


if __name__ == "__main__":
    run(pipeline_name="advanced_rag", pipeline="advanced")
