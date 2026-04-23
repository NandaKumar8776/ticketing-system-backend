"""
Issue Support — RAG Pipeline
Streamlit frontend for the agentic RAG pipeline.

Communicates with the FastAPI backend exclusively via HTTP.
No LangGraph or backend modules are imported here.
"""

import os
import uuid
from datetime import datetime
from typing import Optional

import httpx
import streamlit as st

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

API_URL: str = os.environ.get("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT: float = 60.0  # seconds — RAG pipelines can be slow


# ──────────────────────────────────────────────
# Page Config  (must be the very first Streamlit call)
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Issue Support — RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Tighten the default top padding */
    .block-container { padding-top: 1.5rem; }

    /* Route badges */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-rag  { background: #1a6c3a; color: #d6f5e3; }
    .badge-llm  { background: #1a3d6c; color: #d6eaf5; }

    /* Health indicator dots */
    .dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    .dot-green { background-color: #2ecc71; }
    .dot-red   { background-color: #e74c3c; }

    /* Source cards inside expanders */
    .source-card {
        background: #1e1e2e;
        border-left: 3px solid #4a90d9;
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        line-height: 1.5;
        color: #e8eaf0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Session-state initialisation
# ──────────────────────────────────────────────

def _init_session_state() -> None:
    """Ensure all session-state keys exist on first run."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str, "meta": dict|None}
        st.session_state.messages = []
    if "backend_healthy" not in st.session_state:
        st.session_state.backend_healthy = False


_init_session_state()


# ──────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────

def _get_health() -> Optional[dict]:
    """GET /health — returns dict or None on failure."""
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _get_metrics() -> Optional[dict]:
    """GET /metrics — returns dict or None on failure."""
    try:
        resp = httpx.get(f"{API_URL}/metrics", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _post_chat(query: str, session_id: str) -> dict:
    """
    POST /chat — raises on HTTP or connection errors.
    Returns the parsed JSON dict on success.
    """
    resp = httpx.post(
        f"{API_URL}/chat",
        json={"query": query, "session_id": session_id},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.title("RAG Pipeline")
        st.divider()

        # --- Health Status ---
        st.subheader("Backend Status")
        health = _get_health()
        if health:
            st.session_state.backend_healthy = True
            st.markdown(
                '<span class="dot dot-green"></span> **Online**',
                unsafe_allow_html=True,
            )
            st.caption(f"Version: {health.get('version', 'unknown')}")
        else:
            st.session_state.backend_healthy = False
            st.markdown(
                '<span class="dot dot-red"></span> **Offline**',
                unsafe_allow_html=True,
            )
            st.caption(f"Endpoint: `{API_URL}`")

        st.divider()

        # --- Session ---
        st.subheader("Session")
        st.code(st.session_state.session_id, language=None)
        if st.button("New Session", use_container_width=True, type="secondary"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # --- Metrics ---
        st.subheader("Pipeline Metrics")
        metrics = _get_metrics()
        if metrics:
            col1, col2 = st.columns(2)
            col1.metric("Total Requests", metrics.get("total_requests", 0))

            avg_lat = metrics.get("avg_latency_ms")
            col2.metric(
                "Avg Latency",
                f"{avg_lat:.0f} ms" if avg_lat is not None else "—",
            )

            col3, col4 = st.columns(2)
            col3.metric("RAG Routes", metrics.get("rag_route_count", 0))
            col4.metric("LLM Routes", metrics.get("llm_route_count", 0))

            avg_eval = metrics.get("avg_eval_score")
            st.metric(
                "Avg Eval Score",
                f"{avg_eval:.1f} / 10" if avg_eval is not None else "—",
            )
        else:
            st.caption("Metrics unavailable — is the backend running?")

        st.divider()
        st.caption(f"API: `{API_URL}`")
        st.caption(f"Session started: {datetime.now().strftime('%H:%M:%S')}")


# ──────────────────────────────────────────────
# Response metadata expander
# ──────────────────────────────────────────────

def _render_response_meta(meta: dict) -> None:
    """Render the collapsible metadata block beneath an assistant message."""
    route = meta.get("route", "LLM")
    eval_score = meta.get("eval_score")
    latency_ms = meta.get("latency_ms", 0.0)
    sources = meta.get("sources", [])
    num_sources = meta.get("num_sources", 0)

    badge_class = "badge-rag" if route == "RAG" else "badge-llm"
    badge_html = f'<span class="badge {badge_class}">{route}</span>'

    with st.expander("Response details", expanded=False):
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown("**Route**")
            st.markdown(badge_html, unsafe_allow_html=True)

        with col2:
            st.markdown("**Eval Score**")
            if eval_score is not None:
                st.markdown(f"{eval_score:.1f} / 10")
            else:
                st.markdown("—")

        with col3:
            st.markdown("**Latency**")
            st.markdown(f"{latency_ms:.0f} ms")

        # Sources — only shown when the RAG route was taken
        if route == "RAG" and sources:
            st.divider()
            st.markdown(f"**Retrieved Sources** ({num_sources})")
            for i, src in enumerate(sources, start=1):
                page_label = f"p.{src['page']}" if src.get("page") is not None else "—"
                score_label = f"{src['score']:.3f}" if src.get("score") is not None else "—"
                header = f"Source {i} · page {page_label} · reranker score {score_label}"
                snippet = src.get("content", "")[:400]
                if len(src.get("content", "")) > 400:
                    snippet += "…"
                st.markdown(
                    f'<div class="source-card">'
                    f'<strong>{header}</strong><br>{snippet}'
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────
# Main chat area
# ──────────────────────────────────────────────

def _render_chat() -> None:
    st.title("Issue Support")
    st.caption("Powered by an agentic RAG pipeline — hybrid BM25 + vector search, cross-encoder re-ranking, LLM-as-judge evaluation.")

    # Replay existing conversation from session state
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                _render_response_meta(msg["meta"])

    # --- Input ---
    user_input = st.chat_input(
        "Ask a question about your issues…",
        disabled=not st.session_state.backend_healthy,
    )

    if not st.session_state.backend_healthy and not user_input:
        st.warning(
            "The backend is not reachable at `{url}`. "
            "Start the FastAPI server with `uvicorn api:app --reload` and refresh this page.".format(
                url=API_URL
            )
        )

    if user_input:
        # Display user turn immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append(
            {"role": "user", "content": user_input, "meta": None}
        )

        # Call the backend and display the assistant turn
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    data = _post_chat(
                        query=user_input,
                        session_id=st.session_state.session_id,
                    )
                except httpx.ConnectError:
                    st.error(
                        f"Cannot connect to the backend at `{API_URL}`. "
                        "Please start the FastAPI server and try again."
                    )
                    return
                except httpx.TimeoutException:
                    st.error(
                        f"The request timed out after {REQUEST_TIMEOUT:.0f} s. "
                        "The pipeline may be under heavy load — please retry."
                    )
                    return
                except httpx.HTTPStatusError as exc:
                    st.error(
                        f"Backend returned HTTP {exc.response.status_code}: "
                        f"{exc.response.text[:300]}"
                    )
                    return
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
                    return

            # Render the answer
            answer = data.get("answer", "*(No answer returned)*")
            st.markdown(answer)

            # Build meta dict for the expander
            meta = {
                "route": data.get("route", "LLM"),
                "eval_score": data.get("eval_score"),
                "latency_ms": data.get("latency_ms", 0.0),
                "num_sources": data.get("num_sources", 0),
                "sources": data.get("sources", []),
                "top_rag_score": data.get("top_rag_score"),
            }
            _render_response_meta(meta)

        # Persist assistant turn
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "meta": meta}
        )

        # Sync the session_id the server assigned (it may have been auto-generated)
        if data.get("session_id"):
            st.session_state.session_id = data["session_id"]


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> None:
    _render_sidebar()
    _render_chat()


main()
