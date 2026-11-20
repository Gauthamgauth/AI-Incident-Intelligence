"""
FastAPI backend wrapping the RAG pipelines (naive + advanced) as HTTP endpoints.
Protected by API key auth + rate limiting.
Run with: uvicorn src.api.main:app --reload
"""

import os
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "agents"))
sys.path.append(str(PROJECT_ROOT / "src" / "retrieval"))

load_dotenv(PROJECT_ROOT / ".env")

from generate_answer import generate_answer as generate_naive  # noqa: E402
from retriever import retrieve  # noqa: E402

APP_API_KEY = os.getenv("APP_API_KEY")


# Rate limiter setup

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Incident Intelligence API", version="1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# API key auth setup

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(api_key_header)):
    if not APP_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: no API key set")
    if key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


class AskRequest(BaseModel):
    query: str
    pipeline: Literal["naive", "advanced"] = "advanced"
    n_results: int = 5


class AskResponse(BaseModel):
    query: str
    pipeline: str
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")
def ask(request: Request, body: AskRequest, api_key: str = Security(verify_api_key)):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        if body.pipeline == "naive":
            results = retrieve(body.query, body.n_results)
            answer = generate_naive(body.query, body.n_results)
        else:
            # Lazy import — only loads the heavy reranker model if advanced is actually requested
            from generate_answer_advanced import generate_answer as generate_advanced
            from hybrid_retriever import advanced_retrieve
            results = advanced_retrieve(body.query, body.n_results)
            answer = generate_advanced(body.query, body.n_results)

        sources = [
            {
                "source": r["metadata"]["source"],
                "title": r["metadata"]["title"],
                "impact": r["metadata"]["impact"],
                "status": r["metadata"]["status"],
                "shortlink": r["metadata"]["shortlink"],
            }
            for r in results
        ]

        return AskResponse(
            query=body.query,
            pipeline=body.pipeline,
            answer=answer,
            sources=sources,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))














