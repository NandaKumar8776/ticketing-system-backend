"""
FastAPI REST API for the Issue Support RAG System.

Provides endpoints for:
- /chat       : Conversational RAG-powered Q&A
- /ingest     : Upload a PDF and add it to the retrieval knowledge base
- /health     : Readiness probe — checks all backend dependencies
- /metrics    : Aggregated pipeline performance metrics
"""

import os
import shutil
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Rate limiter ---
limiter = Limiter(key_func=get_remote_address)


# ──────────────────────────────────────────────
# Session store helpers  (#2, #8)
# ──────────────────────────────────────────────

_MAX_SESSIONS   = int(os.getenv("MAX_SESSIONS", "1000"))
_MAX_MSG_WINDOW = int(os.getenv("MAX_MSG_WINDOW", "20"))   # messages kept per session

# {session_id: {"messages": [...], "last_access": float}}
_sessions: dict[str, dict] = {}


def _get_session(session_id: str) -> list:
    """Return the message list for a session, creating it if needed."""
    now = time.time()
    if session_id not in _sessions:
        # Evict the oldest session if we're at capacity
        if len(_sessions) >= _MAX_SESSIONS:
            oldest = min(_sessions, key=lambda sid: _sessions[sid]["last_access"])
            del _sessions[oldest]
        _sessions[session_id] = {"messages": [], "last_access": now}
    _sessions[session_id]["last_access"] = now
    return _sessions[session_id]["messages"]


def _trim_session(messages: list) -> list:
    """Keep only the last _MAX_MSG_WINDOW messages to stay within context limits."""
    if len(messages) > _MAX_MSG_WINDOW:
        del messages[: len(messages) - _MAX_MSG_WINDOW]
    return messages


# ──────────────────────────────────────────────
# Lifespan: startup / shutdown  (#16)
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize environment, vector store, and BM25 on startup."""
    logger.info("Starting Issue Support RAG API...")

    import config.env_setup  # noqa: F401

    from tools.document_loader import initialize_retrievers
    initialize_retrievers()

    # Pre-load the cross-encoder so the first /chat request doesn't pay model load time
    from tools.reranker import get_reranker
    get_reranker()

    logger.info("All retrievers initialized. API is ready.")
    yield

    # Graceful shutdown: close persistent connections
    logger.info("Shutting down Issue Support RAG API...")
    try:
        import utils.gcs_store as _gcs_module
        if _gcs_module._gcs_client is not None:
            _gcs_module._gcs_client.close()
            logger.info("GCS client closed.")
    except Exception as e:
        logger.warning(f"Error closing GCS client: {e}")


# ──────────────────────────────────────────────
# App & Middleware
# ──────────────────────────────────────────────

app = FastAPI(
    title="Issue Support RAG API",
    description=(
        "Agentic RAG pipeline with LangGraph, hybrid BM25 + vector search, "
        "cross-encoder re-ranking, and LLM-as-judge evaluation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: restrict to known origins in production via ALLOWED_ORIGINS env var.  (#10)
# Defaults to * so local dev and the demo frontend work without configuration.
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# API Key Auth Middleware
# ──────────────────────────────────────────────

_DEMO_API_KEY = os.getenv("DEMO_API_KEY", "")
_PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Require X-API-Key header when DEMO_API_KEY is set."""
    if _DEMO_API_KEY and request.url.path not in _PUBLIC_PATHS:
        key = request.headers.get("X-API-Key", "")
        if key != _DEMO_API_KEY:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid or missing API key. Pass your key as X-API-Key header."},
            )
    return await call_next(request)


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SourceDocument(BaseModel):
    content: str
    page: Optional[int] = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    route: str
    top_rag_score: Optional[float] = None
    num_sources: int = 0
    sources: list[SourceDocument] = Field(default_factory=list)
    latency_ms: float
    eval_score: Optional[float] = None
    guardrail_triggered: bool = False
    guardrail_reason: Optional[str] = None


class IngestResponse(BaseModel):
    filename: str
    num_chunks: int
    total_corpus_chunks: int
    success: bool
    gcs_persisted: bool = False
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    checks: dict


class MetricsSummary(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    rag_route_count: int
    llm_route_count: int
    avg_rag_score: Optional[float]
    avg_eval_score: Optional[float]


# ──────────────────────────────────────────────
# In-memory metrics store  (#15)
# ──────────────────────────────────────────────

_metrics_store: list[dict] = []


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(os.getenv("CHAT_RATE_LIMIT", "30/minute"))   # #11
async def chat(request: Request, body: ChatRequest):
    """
    Process a user query through the agentic RAG pipeline.

    Steps:
    1. Router scores the query against the knowledge base (hybrid BM25 + vector).
    2. If relevant → RAG node generates an answer with retrieved context.
       If not   → generic LLM generates a conversational response.
    3. Evaluator node scores the response quality.
    4. Metrics are logged per request (including failures).
    """
    from graph.workflow import app as langgraph_app

    start = time.perf_counter()
    session_messages = _get_session(body.session_id)
    session_messages.append({"role": "user", "content": body.query})
    _trim_session(session_messages)

    try:
        result = langgraph_app.invoke({"messages": session_messages})
    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        # Log failed requests to metrics so error rates are visible  (#15)
        _metrics_store.append({
            "session_id": body.session_id,
            "route": "ERROR",
            "latency_ms": latency_ms,
            "eval_score": None,
            "top_rag_score": None,
            "guardrail_triggered": False,
            "error": str(e),
            "timestamp": time.time(),
        })
        logger.error(f"LangGraph invocation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    last_message = result.get("messages", [None])[-1]
    if last_message is None:
        raise HTTPException(status_code=500, detail="No response generated")

    answer = getattr(last_message, "content", "")
    if not answer and isinstance(last_message, dict):
        answer = last_message.get("content", "")

    session_messages.append({"role": "assistant", "content": answer})

    guardrail_triggered = result.get("guardrail_triggered", False)
    guardrail_reason    = result.get("guardrail_reason", None)

    if guardrail_triggered:
        route_label = "BLOCKED"
    else:
        category = result.get("category", "Not Related")
        route_label = "RAG" if "rag" in category.lower() else "LLM"

    context_docs = result.get("context", []) or []
    sources = []
    top_rag_score = result.get("top_rag_score")

    for doc in context_docs:
        page  = doc.metadata.get("page") if hasattr(doc, "metadata") else None
        content = getattr(doc, "page_content", str(doc))
        score = doc.metadata.get("rerank_score") if hasattr(doc, "metadata") else None
        sources.append(SourceDocument(content=content[:500], page=page, score=score))

    eval_score = result.get("eval_score")

    metrics_record = {
        "session_id": body.session_id,
        "route": route_label,
        "top_rag_score": top_rag_score,
        "num_sources": len(sources),
        "latency_ms": latency_ms,
        "eval_score": eval_score,
        "query_length": len(body.query),
        "guardrail_triggered": guardrail_triggered,
        "guardrail_reason": guardrail_reason,
        "timestamp": time.time(),
    }
    _metrics_store.append(metrics_record)

    try:
        from utils.metrics import log_metrics
        log_metrics(metrics_record)
    except ImportError:
        pass

    return ChatResponse(
        answer=answer,
        session_id=body.session_id,
        route=route_label,
        top_rag_score=top_rag_score,
        num_sources=len(sources),
        sources=sources,
        latency_ms=latency_ms,
        eval_score=eval_score,
        guardrail_triggered=guardrail_triggered,
        guardrail_reason=guardrail_reason,
    )


@app.post("/ingest", response_model=IngestResponse)
@limiter.limit(os.getenv("INGEST_RATE_LIMIT", "10/minute"))  # #11
async def ingest(request: Request, file: UploadFile = File(...)):
    """Upload a PDF and add it to the retrieval knowledge base."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    upload_dir = os.getenv("UPLOAD_DIR", "/app/data/uploads")
    os.makedirs(upload_dir, exist_ok=True)
    dest_path = os.path.join(upload_dir, file.filename)

    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    from tools.document_loader import ingest_pdf
    result = ingest_pdf(dest_path)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return IngestResponse(
        filename=result["filename"],
        num_chunks=result["num_chunks"],
        total_corpus_chunks=result["total_corpus_chunks"],
        success=True,
        gcs_persisted=result.get("gcs_persisted", False),
        message=f"Successfully ingested '{result['filename']}' — {result['num_chunks']} chunks added.",
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Readiness probe — checks all backend dependencies."""  # #9
    checks = {}

    # Milvus
    try:
        from memory.vector_store import flat_milvus_vector_store
        checks["milvus"] = "ok" if flat_milvus_vector_store is not None else "unavailable"
    except Exception:
        checks["milvus"] = "error"

    # Retrievers
    try:
        from tools.document_loader import BM25_retriever, vector_store_retriever
        checks["bm25"]         = "ok"    if BM25_retriever         is not None else "empty"
        checks["vector_store"] = "ok"    if vector_store_retriever is not None else "empty"
    except Exception:
        checks["bm25"] = checks["vector_store"] = "error"

    # GCS
    from utils.gcs_store import is_configured
    checks["gcs"] = "configured" if is_configured() else "not_configured"

    # Overall status: degraded only if a hard dependency is in error
    hard_fail = any(v == "error" for v in checks.values())
    status = "degraded" if hard_fail else "healthy"

    return HealthResponse(status=status, version="1.0.0", checks=checks)


@app.get("/metrics", response_model=MetricsSummary)
async def metrics():
    """Return aggregated pipeline performance metrics."""
    if not _metrics_store:
        return MetricsSummary(
            total_requests=0, successful_requests=0, failed_requests=0,
            avg_latency_ms=0.0, rag_route_count=0, llm_route_count=0,
            avg_rag_score=None, avg_eval_score=None,
        )

    total     = len(_metrics_store)
    failed    = sum(1 for m in _metrics_store if m.get("route") == "ERROR")
    successful = total - failed

    avg_latency = sum(m["latency_ms"] for m in _metrics_store) / total
    rag_count   = sum(1 for m in _metrics_store if m.get("route") == "RAG")
    llm_count   = sum(1 for m in _metrics_store if m.get("route") == "LLM")

    rag_scores  = [m["top_rag_score"] for m in _metrics_store if m.get("top_rag_score") is not None]
    eval_scores = [m["eval_score"]     for m in _metrics_store if m.get("eval_score")     is not None]

    return MetricsSummary(
        total_requests=total,
        successful_requests=successful,
        failed_requests=failed,
        avg_latency_ms=round(avg_latency, 2),
        rag_route_count=rag_count,
        llm_route_count=llm_count,
        avg_rag_score=round(sum(rag_scores)  / len(rag_scores),  4) if rag_scores  else None,
        avg_eval_score=round(sum(eval_scores) / len(eval_scores), 2) if eval_scores else None,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
