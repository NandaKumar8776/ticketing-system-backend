"""
FastAPI REST API for the Issue Support RAG System.

Provides endpoints for:
- /chat       : Conversational RAG-powered Q&A
- /ingest     : Upload a PDF and add it to the retrieval knowledge base
- /health     : Liveness probe for deployment readiness
- /metrics    : Aggregated pipeline performance metrics
"""

import os
import shutil
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Lifespan: startup / shutdown
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize environment, vector store, and BM25 on startup."""
    logger.info("Starting Issue Support RAG API...")

    # 1. Load environment variables & configure tracing
    import config.env_setup  # noqa: F401

    # 2. Initialize document retrievers (Milvus + BM25)
    from tools.document_loader import initialize_retrievers
    initialize_retrievers()

    logger.info("All retrievers initialized. API is ready.")
    yield
    logger.info("Shutting down Issue Support RAG API.")


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming chat request."""
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session ID for conversation continuity",
    )


class SourceDocument(BaseModel):
    """A retrieved source chunk returned with the answer."""
    content: str
    page: Optional[int] = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    """Structured response from the RAG pipeline."""
    answer: str = Field(..., description="Generated answer")
    session_id: str = Field(..., description="Session ID for this conversation")
    route: str = Field(..., description="Pipeline route taken: 'RAG', 'LLM', or 'BLOCKED'")
    top_rag_score: Optional[float] = Field(None, description="Highest retrieval relevance score")
    num_sources: int = Field(0, description="Number of source documents used")
    sources: list[SourceDocument] = Field(default_factory=list, description="Retrieved source chunks")
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds")
    eval_score: Optional[float] = Field(None, description="LLM-as-judge evaluation score (0-10)")
    guardrail_triggered: bool = Field(False, description="True if the request was blocked by guardrails")
    guardrail_reason: Optional[str] = Field(None, description="Reason for guardrail block if triggered")


class IngestResponse(BaseModel):
    """Response from the /ingest endpoint."""
    filename: str = Field(..., description="Name of the ingested file")
    num_chunks: int = Field(..., description="Number of chunks indexed")
    total_corpus_chunks: int = Field(..., description="Total chunks in the knowledge base after ingestion")
    success: bool = Field(..., description="Whether ingestion succeeded")
    message: str = Field(..., description="Human-readable status message")


class HealthResponse(BaseModel):
    status: str
    version: str


class MetricsSummary(BaseModel):
    total_requests: int
    avg_latency_ms: float
    rag_route_count: int
    llm_route_count: int
    avg_rag_score: Optional[float]
    avg_eval_score: Optional[float]


# ──────────────────────────────────────────────
# In-memory session store & metrics counters
# ──────────────────────────────────────────────

_sessions: dict[str, list[dict]] = {}
_metrics_store: list[dict] = []


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a user query through the agentic RAG pipeline.

    Steps:
    1. Router scores the query against the knowledge base (hybrid BM25 + vector).
    2. If relevant → RAG node generates an answer with retrieved context.
       If not   → generic LLM generates a conversational response.
    3. (Optional) Evaluator node scores the response quality on a rubric.
    4. Metrics are logged per request.
    """
    from graph.workflow import app as langgraph_app

    start = time.perf_counter()

    # Retrieve or create session history
    session_messages = _sessions.setdefault(request.session_id, [])
    session_messages.append({"role": "user", "content": request.query})

    try:
        # Invoke the LangGraph workflow
        result = langgraph_app.invoke({"messages": session_messages})
    except Exception as e:
        logger.error(f"LangGraph invocation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    # Extract answer from result
    last_message = result.get("messages", [None])[-1]
    if last_message is None:
        raise HTTPException(status_code=500, detail="No response generated")

    answer = getattr(last_message, "content", "")
    if not answer and isinstance(last_message, dict):
        answer = last_message.get("content", "")

    # Save assistant response to session
    session_messages.append({"role": "assistant", "content": answer})

    # Extract guardrail and routing metadata from state
    guardrail_triggered = result.get("guardrail_triggered", False)
    guardrail_reason = result.get("guardrail_reason", None)

    if guardrail_triggered:
        route_label = "BLOCKED"
    else:
        category = result.get("category", "Not Related")
        route_label = "RAG" if "rag" in category.lower() else "LLM"

    # Extract source documents & scores if available
    context_docs = result.get("context", []) or []
    sources = []
    top_rag_score = None

    for doc in context_docs:
        page = None
        if hasattr(doc, "metadata"):
            page = doc.metadata.get("page")
        content = getattr(doc, "page_content", str(doc))
        score = doc.metadata.get("rerank_score") if hasattr(doc, "metadata") else None
        sources.append(SourceDocument(content=content[:500], page=page, score=score))

    # Extract eval score if present (will be populated once evaluator is wired in)
    eval_score = result.get("eval_score")

    # Log metrics
    metrics_record = {
        "session_id": request.session_id,
        "route": route_label,
        "top_rag_score": top_rag_score,
        "num_sources": len(sources),
        "latency_ms": latency_ms,
        "eval_score": eval_score,
        "query_length": len(request.query),
        "guardrail_triggered": guardrail_triggered,
        "guardrail_reason": guardrail_reason,
        "timestamp": time.time(),
    }
    _metrics_store.append(metrics_record)

    # Also log to file-based metrics
    try:
        from utils.metrics import log_metrics
        log_metrics(metrics_record)
    except ImportError:
        pass  # metrics module not yet created

    return ChatResponse(
        answer=answer,
        session_id=request.session_id,
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
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF and add it to the retrieval knowledge base.

    The file is chunked, embedded, and indexed into both Milvus (dense) and
    BM25 (sparse) retrievers without restarting the server. All subsequent
    /chat requests will include the new document in retrieval.

    Accepts: multipart/form-data with a PDF file field named 'file'.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save upload to a temp location inside the container data dir
    upload_dir = os.getenv("UPLOAD_DIR", "/app/data/uploads")
    os.makedirs(upload_dir, exist_ok=True)
    dest_path = os.path.join(upload_dir, file.filename)

    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    from tools.document_loader import ingest_pdf, _all_docs
    result = ingest_pdf(dest_path)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])

    from tools.document_loader import _all_docs as corpus
    return IngestResponse(
        filename=result["filename"],
        num_chunks=result["num_chunks"],
        total_corpus_chunks=len(corpus),
        success=True,
        message=f"Successfully ingested '{result['filename']}' — {result['num_chunks']} chunks added.",
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe for container orchestration."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/metrics", response_model=MetricsSummary)
async def metrics():
    """Return aggregated pipeline performance metrics."""
    if not _metrics_store:
        return MetricsSummary(
            total_requests=0,
            avg_latency_ms=0.0,
            rag_route_count=0,
            llm_route_count=0,
            avg_rag_score=None,
            avg_eval_score=None,
        )

    total = len(_metrics_store)
    avg_latency = sum(m["latency_ms"] for m in _metrics_store) / total
    rag_count = sum(1 for m in _metrics_store if m["route"] == "RAG")
    llm_count = total - rag_count

    rag_scores = [m["top_rag_score"] for m in _metrics_store if m["top_rag_score"] is not None]
    avg_rag = sum(rag_scores) / len(rag_scores) if rag_scores else None

    eval_scores = [m["eval_score"] for m in _metrics_store if m["eval_score"] is not None]
    avg_eval = sum(eval_scores) / len(eval_scores) if eval_scores else None

    return MetricsSummary(
        total_requests=total,
        avg_latency_ms=round(avg_latency, 2),
        rag_route_count=rag_count,
        llm_route_count=llm_count,
        avg_rag_score=round(avg_rag, 4) if avg_rag else None,
        avg_eval_score=round(avg_eval, 2) if avg_eval else None,
    )


# ──────────────────────────────────────────────
# Run directly: uvicorn api:app --reload
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
