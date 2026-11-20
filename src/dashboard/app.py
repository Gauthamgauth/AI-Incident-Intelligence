"""
Streamlit dashboard: a simple chat-style UI for querying the Incident
Intelligence RAG system via the FastAPI backend.
"""

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

API_URL = "http://127.0.0.1:8000/ask"
APP_API_KEY = os.getenv("APP_API_KEY")

st.set_page_config(page_title="Incident Intelligence Assistant", page_icon="🔍", layout="centered")

st.title("🔍 AI-Powered Incident Intelligence Assistant")
st.caption("Ask questions about past GitHub, Cloudflare, OpenAI & Anthropic incidents using AI-powered retrieval")

with st.sidebar:
    st.header("Settings")
    pipeline = st.radio(
        "Pipeline",
        options=["advanced", "naive"],
        format_func=lambda x: "Advanced (Hybrid + Rerank)" if x == "advanced" else "Naive (Vector only)",
    )
    n_results = st.slider("Number of sources", min_value=1, max_value=10, value=5)

query = st.text_input("Ask a question:", placeholder="what caused API outages recently")

if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Retrieving and generating answer..."):
        try:
            response = requests.post(
                API_URL,
                json={"query": query, "pipeline": pipeline, "n_results": n_results},
                headers={"X-API-Key": APP_API_KEY},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

            st.subheader(f"Sources ({len(data['sources'])})")
            for src in data["sources"]:
                impact_color = {"critical": "🔴", "major": "🟠", "minor": "🟡", "none": "🟢"}.get(src["impact"], "⚪")
                with st.expander(f"{impact_color} [{src['source']}] {src['title']}"):
                    st.write(f"**Status:** {src['status']}")
                    st.write(f"**Impact:** {src['impact']}")
                    if src["shortlink"]:
                        st.write(f"**Link:** {src['shortlink']}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure the FastAPI server is running (uvicorn src.api.main:app --reload).")
        except Exception as e:
            st.error(f"Error: {e}")