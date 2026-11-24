# AI-Powered Engineering Incident Intelligence & Root Cause Analysis System

A RAG (Retrieval-Augmented Generation) system that ingests real engineering incident data from public status pages, stores it as searchable embeddings, and answers root-cause questions with grounded, cited answers — comparing a naive retrieval baseline against a hardened, hybrid-search pipeline.

**Live demo:**
- Dashboard: ai-incident-intelligence.streamlit.app
- API: ai-incident-intelligence-fw8d.onrender.com
- Code: github.com/Gauthamgauth/AI-Incident-Intelligence

---

## What this project actually does

Give it a question like *"What caused the most recent GitHub outage?"* and it retrieves the most relevant real incident reports (scraped from public status pages), then generates a grounded answer that cites its sources — instead of hallucinating from general knowledge.

This is **not** a fine-tuned or custom-trained model. It uses pretrained components (OpenAI's embedding model, GPT-4o-mini) orchestrated into a retrieval pipeline. The engineering work here is in the *pipeline design*, not model training — data engineering, retrieval architecture, evaluation, and production hardening.

---

## Architecture / Pipeline

```
Status Page APIs (GitHub, Cloudflare, OpenAI, Anthropic)
        │
        ▼
Ingestion & Normalization  →  one unified schema across 4 sources
        │
        ▼
PostgreSQL  →  175 cleaned, deduplicated incidents stored
        │
        ▼
Chunking & Embedding  →  290 chunks, OpenAI text-embedding-3-small
        │
        ▼
ChromaDB  →  persistent vector store
        │
        ▼
   ┌────┴────┐
   │         │
Naive RAG   Advanced RAG
(vector      (BM25 + vector hybrid search
 search      + cross-encoder reranking)
 only)
   │         │
   └────┬────┘
        ▼
LLM Generation (gpt-4o-mini)  →  grounded answer + cited sources
        │
        ▼
LLM-as-Judge Evaluation  →  Retrieval Relevance, Faithfulness, Answer Relevance
        │
        ▼
FastAPI backend (+ auth, rate limiting)  →  Streamlit dashboard
```

---

## Data Engineering

- Scraped 4 public status-page APIs (GitHub, Cloudflare, OpenAI, Anthropic) using the Atlassian Statuspage format
- Slack was deliberately dropped — its API uses a different, non-Statuspage schema that would have broken the unified data model
- Normalized all 4 sources into one schema: `id, source, title, impact, status, created_at, resolved_at, shortlink, components, updates_text`
- Deduplicated and cleaned 175 incidents total, saved to `data/processed/incidents_clean.json`
- Loaded into PostgreSQL 18 (local, via pgAdmin 4) as the structured source of truth

---

## The Core Improvement: Naive RAG → Advanced RAG

This is the real "did it get better, and how do you know" story of the project — the same discipline of measuring before claiming improvement that matters in any ML/AI system, trained model or not.

### Naive RAG (baseline)
Plain vector similarity search against ChromaDB, straight into the LLM.

| Metric | Score |
|---|---|
| Retrieval Relevance | 5.0 / 5 |
| Faithfulness | 4.6 / 5 |
| Answer Relevance | 4.3 / 5 |

### Advanced RAG (hardened)
Added **hybrid retrieval** (BM25 keyword search + vector search combined) and **cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) on top of the candidates before they reach the LLM.

| Metric | Naive | Advanced | Change |
|---|---|---|---|
| Retrieval Relevance | 5.0 / 5 | 5.0 / 5 | — |
| Faithfulness | 4.6 / 5 | **5.0 / 5** | ↑ |
| Answer Relevance | 4.3 / 5 | **4.6 / 5** | ↑ |

**Why this matters:** Faithfulness measures whether the answer is actually grounded in the retrieved sources rather than drifting into plausible-sounding invention. The improvement here is the project's strongest evidence that hybrid search + reranking produces more trustworthy answers, not just "different" ones — validated with an LLM-as-judge evaluation harness (`src/evaluation/evaluate.py`), not eyeballed.

Evaluation was run via **LLM-as-judge** (gpt-4o-mini scoring against a 10-query test set), with both result sets saved for comparison (`naive_rag_results.json`, `advanced_rag_results.json`).

---

## Production Hardening

- **Observability**: LangSmith tracing via `@traceable` decorators across the retriever, hybrid retriever, and both generation functions — every run is inspectable after the fact, not just via terminal output
- **API security**: API key authentication (`X-API-Key` header) and rate limiting (10 requests/minute via `slowapi`) added to the FastAPI backend before deployment
- **Interface**: A FastAPI backend (`POST /ask` with naive/advanced pipeline choice, `GET /health`) behind a Streamlit dashboard with a pipeline toggle and color-coded, collapsible source cards by impact level

---

## Known Limitations

Being upfront about this rather than hiding it:

- **The advanced pipeline (hybrid search + reranking) hits a 512MB RAM limit on Render's free tier and returns a 502 error in production.** It works correctly and fully locally — this is a hosting-tier resource constraint, not a logic bug. The naive pipeline runs fine in production since it's lighter-weight and loads at startup; the advanced pipeline is lazy-loaded inside the endpoint specifically to try to fit the free-tier memory ceiling, but the reranking model's memory footprint still exceeds it under load.
- **Practical takeaway for demos:** the live naive pipeline is fully demoable end-to-end; the advanced pipeline (the one with the better eval scores) is demoed locally during interviews rather than on the live deployment, since the free hosting tier can't currently support its memory requirements.
- **The honest fix** would be either a paid hosting tier with more RAM, or swapping the cross-encoder reranker for a lighter-weight reranking approach that fits within 512MB — neither was pursued further, since the local-demo workaround was judged sufficient for this project's goal (a fresher portfolio piece) rather than a production SLA.

---

## Tech Stack

Python 3.12.2 · LangChain · LangGraph · PostgreSQL 18 · ChromaDB · OpenAI (embeddings + gpt-4o-mini) · sentence-transformers (cross-encoder reranking) · FastAPI · slowapi · Streamlit · LangSmith · Docker (planned in full roadmap)

---
