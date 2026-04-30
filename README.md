# Production RAG Pipeline — Agentic IT Support Assistant

> Production-grade agentic RAG pipeline built with **LangGraph**, **hybrid BM25 + vector search**, **cross-encoder re-ranking**, **multi-layer guardrails**, and **LLM-as-judge evaluation** — served via **FastAPI**, deployed on **GCP Cloud Run**, backed by **self-hosted Milvus on GKE** with full HNSW indexing, **GCS document persistence**, and an **MCP server** for Claude Desktop integration.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple.svg)](https://github.com/langchain-ai/langgraph)
[![MCP](https://img.shields.io/badge/MCP-1.0-orange.svg)](https://modelcontextprotocol.io)
[![Cloud Run](https://img.shields.io/badge/GCP-Cloud%20Run-4285F4.svg)](https://cloud.google.com/run)

---

## Live Demo

**API:** `https://it-support-rag-c72zrk22aa-uc.a.run.app`

```bash
# Chat
curl -X POST https://it-support-rag-c72zrk22aa-uc.a.run.app/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <secret_key>" \
  -d '{"query": "My PC fan is not turning on"}'

# Ingest a PDF (persists to GCS — survives container restarts)
curl -X POST https://it-support-rag-c72zrk22aa-uc.a.run.app/ingest \
  -H "X-API-Key: <secret_key>" \
  -F "file=@your-document.pdf"

# Health (public)
curl https://it-support-rag-c72zrk22aa-uc.a.run.app/health
```

All endpoints except `/health`, `/docs`, and `/redoc` require `X-API-Key` header. Contact for access.

---

## Architecture

### Request Pipeline

```
User Query
    │
    ▼
[FastAPI /chat]   ◄─── X-API-Key auth ─── Rate limiter (30 req/min)
    │
    ▼
┌─────────────────────┐
│   Guardrails Node   │  ← 4-layer: prompt injection → jailbreak → PII → LLM classifier
└──────────┬──────────┘
           │
      safe / blocked?
           │
     ▼ (safe)      ▼ (blocked) ──► Refusal message
┌──────────────────┐
│   Router Node    │  ← Hybrid BM25 + Milvus HNSW retrieval
│  (Score-gated)   │    + Cross-encoder re-ranking (ms-marco-MiniLM)
└────────┬─────────┘
         │
    score ≥ threshold?
    /              \
   ▼                ▼
[RAG Node]      [LLM Node]
(Groq + context) (Groq, general)
    \                /
     ▼              ▼
  [Evaluator Node]       ← LLM-as-Judge (4-dimension rubric, 0–10)
         │
    eval_score + answer
         │
         ▼
   JSON Response + Langfuse trace + /metrics
```

### Cloud Infrastructure

```
┌─────────────────────────────────────────────────────────────────┐
│                        GCP Project                              │
│                                                                 │
│  ┌──────────────────┐        ┌──────────────────────────────┐  │
│  │   Cloud Run      │        │   GKE Cluster (us-central1)  │  │
│  │  (it-support-rag)│        │                              │  │
│  │                  │  VPC   │  ┌────────────────────────┐  │  │
│  │  FastAPI +       ├───────►│  │  Milvus Standalone Pod  │  │  │
│  │  LangGraph       │Connector│  │  HNSW index, L2 metric │  │  │
│  │  pipeline        │        │  │  port 19530             │  │  │
│  └────────┬─────────┘        │  └────────────┬───────────┘  │  │
│           │                  │               │               │  │
│           │                  │  ┌────────────▼───────────┐  │  │
│           │                  │  │  Internal LoadBalancer  │  │  │
│           │                  │  │  IP: 10.128.0.7:19530   │  │  │
│           │                  │  └────────────────────────┘  │  │
│           │                  │                              │  │
│           │                  │  Persistent Disks (PVCs):    │  │
│           │                  │  ├─ milvus:      50Gi        │  │
│           │                  │  ├─ minio:       20Gi        │  │
│           │                  │  └─ etcd:        10Gi        │  │
│           │                  └──────────────────────────────┘  │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐                                          │
│  │   Cloud Storage  │                                          │
│  │  (GCS Bucket)    │                                          │
│  │  /documents/     │  ← PDFs uploaded on /ingest             │
│  │  /cache/ (DVC)   │  ← Downloaded on container cold start   │
│  └──────────────────┘                                          │
│                                                                 │
│  ┌──────────────────┐   ┌──────────────────┐                  │
│  │  Artifact        │   │  Secret Manager  │                  │
│  │  Registry        │   │  groq-api-key    │                  │
│  │  (Docker images) │   │  demo-api-key    │                  │
│  └──────────────────┘   └──────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### Document Persistence (GCS)

```
POST /ingest (PDF)
    │
    ├─► Chunked → BM25 index + Milvus HNSW (current container)
    └─► Uploaded to gs://ticket-support-01-dvc/documents/  ← persisted

Container cold start
    │
    └─► Downloads all PDFs from GCS → re-indexes into BM25 + Milvus → ready
        (knowledge base survives container restarts and new deployments)
```

### CI/CD Pipeline

```
git push (main)
    │
    ▼
Cloud Build trigger
    │
    ├─► docker build issue_support/
    ├─► docker push → Artifact Registry
    └─► gcloud run deploy it-support-rag
            ├─ VPC connector: milvus-connector → GKE
            ├─ APP_MILVUS_URI: http://10.128.0.7:19530
            └─ GCS_BUCKET: ticket-support-01-dvc
```

---

## Key Technical Features

| Feature | Implementation |
|---|---|
| **Multi-Layer Guardrails** | 4-stage: prompt injection → jailbreak → PII → LLM abuse classifier |
| **Hybrid Retrieval** | BM25 (sparse) + Milvus HNSW on GKE (dense) with Reciprocal Rank Fusion |
| **Two-Stage Re-ranking** | Ensemble retrieval → cross-encoder re-ranker (`ms-marco-MiniLM-L-6-v2`) |
| **Score-Gated Routing** | Queries routed to RAG only when top retrieval score exceeds threshold |
| **LLM-as-Judge Evaluation** | 4-dimension rubric: relevance, safety, actionability, completeness (0–10) |
| **Agentic Graph** | LangGraph `StateGraph` with conditional routing and typed state |
| **GCS Document Store** | Uploaded PDFs persisted to GCS — knowledge base survives container restarts |
| **Dynamic PDF Ingestion** | `POST /ingest` — any PDF at runtime, indexed without restart |
| **Multi-Modal PDF** | PyMuPDF + Tesseract OCR + markdown table extraction |
| **BM25 Fallback** | Degrades gracefully to keyword-only retrieval if Milvus is unavailable |
| **Session Management** | Sliding window (20 msg), max 1000 sessions, LRU eviction |
| **Rate Limiting** | 30 req/min on `/chat`, 10 req/min on `/ingest` (configurable) |
| **LLM Resilience** | 30s timeout + 2 retries with backoff on all Groq calls |
| **Observability** | Langfuse traces every node; `/metrics` tracks success/error rates, latency, eval scores |
| **MCP Server** | Exposes `query_it_support` + `get_pipeline_metrics` as tools for Claude Desktop |
| **Data Versioning** | DVC tracks knowledge base PDFs — GCS bucket as remote |

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/chat` | Required | RAG-powered Q&A, session continuity, full pipeline metadata |
| `POST` | `/ingest` | Required | Upload PDF, index into BM25 + Milvus, persist to GCS |
| `GET` | `/health` | Public | Dependency check: Milvus, BM25, vector store, GCS |
| `GET` | `/metrics` | Required | Aggregated latency, route distribution, eval scores, error rate |
| `GET` | `/docs` | Public | Interactive Swagger UI |

### Chat Request / Response

```bash
curl -X POST https://it-support-rag-c72zrk22aa-uc.a.run.app/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <secret_key>" \
  -d '{"query": "My PC wont boot after a Windows update", "optional-session_id": "optional-uuid"}'
```

---

## GCP Deployment (Primary)

The production deployment runs on **Google Cloud Run** connected via VPC to **self-hosted Milvus on GKE**, with images in **Artifact Registry**, secrets in **Secret Manager**, and documents in **GCS**.

### Environment Variables (Cloud Run)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — (Secret Manager) | Groq API key |
| `DEMO_API_KEY` | — (Secret Manager) | X-API-Key value for demo auth |
| `GCS_BUCKET` | `ticket-support-01-dvc` | GCS bucket for document persistence |
| `APP_MILVUS_URI` | `http://10.128.0.7:19530` | GKE Milvus internal LoadBalancer IP |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model for general LLM node |
| `RAG_LLM_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq model for RAG node |
| `RAG_SCORE_THRESHOLD` | `0.35` | Min retrieval score to route to RAG |
| `CHAT_RATE_LIMIT` | `30/minute` | Rate limit for `/chat` |
| `INGEST_RATE_LIMIT` | `10/minute` | Rate limit for `/ingest` |
| `MAX_SESSIONS` | `1000` | Max concurrent session slots |
| `MAX_MSG_WINDOW` | `20` | Messages kept per session (sliding window) |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `MILVUS_COLLECTION_NAME` | `IT_Support_Knowledge_Base` | Milvus collection name |

### Milvus Deployment Options

| Option | `APP_MILVUS_URI` | Notes |
|---|---|---|
| **GKE (production)** | `http://<internal-lb-ip>:19530` | Self-hosted on GKE, HNSW index, persistent PVCs |
| **Zilliz Cloud** | `https://...zillizcloud.com` | Managed — set `ZILLIZ_API_KEY` secret |
| **Local Docker** | `http://localhost:19530` | Local dev with Docker Compose |

---

## Data Versioning (DVC)

The knowledge base PDF is tracked with [DVC](https://dvc.org). Git stores only a tiny `.dvc` pointer (hash + path); the actual file lives in GCS.

**Remote:** `gs://ticket-support-01-dvc/cache`

```bash
# After cloning — restore tracked files locally
dvc pull

# Replace the knowledge base PDF
cp new-manual.pdf data/PC_trouble-shooting.pdf
dvc add data/PC_trouble-shooting.pdf
dvc push       
git add data/PC_trouble-shooting.pdf.dvc && git commit -m "update knowledge base"
git push
# Next Cloud Run deploy will serve the new document automatically
```

### Tracked Artifacts

| File | Why DVC |
|---|---|
| `data/PC_trouble-shooting.pdf` | Knowledge base — swap for a larger corpus without bloating git |
| `scripts/eval_results.json` | Baseline eval snapshot — reproducible by checking out the matching tag |
| `scripts/eval_results_v2.json` | Post-tuning eval snapshot |

---

---

## MCP Server

The pipeline is exposed as an [MCP](https://modelcontextprotocol.io) server so Claude Desktop can call the RAG pipeline as a native tool.

### Tools

| Tool | Description |
|---|---|
| `query_it_support` | Ask a question — runs the full RAG pipeline, returns sourced answer |
| `get_pipeline_metrics` | Aggregated latency, route distribution, eval scores |

### Setup

```bash
# Inspect / test the MCP server (Windows)
npx @modelcontextprotocol/inspector "C:\path\to\python.exe" mcp_server.py
```

Add to Claude Desktop `claude_desktop_config.json`:


---

## Pipeline Performance

Evaluated against a 12-query golden test set (8 PC troubleshooting + 4 off-topic).

| Metric | Value |
|---|---|
| Avg eval score (rubric, 0–10) | **7.88** |
| RAG routing precision | **100%** |
| Overall routing precision | **66.7%** |
| Avg end-to-end latency | **3,094ms** |
| Chunk recall@3 | **100%** |

```bash
python scripts/evaluate.py --output scripts/eval_results.json
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph, conditional edges, typed state) |
| Vector Database | Self-hosted Milvus on GKE (HNSW index, L2 metric, persistent PVCs) |
| Sparse Retrieval | BM25 (rank-bm25) |
| Re-ranking | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| LLM Inference | Groq (Llama-3.3-70b, Llama-4-Scout) — 30s timeout, 2 retries |
| API Framework | FastAPI + Pydantic v2 + slowapi rate limiting |
| Document Storage | Google Cloud Storage (persistent knowledge base) |
| Data Versioning | DVC (GCS remote — `gs://ticket-support-01-dvc/cache`) |
| Guardrails | Regex + LLM classifier (custom, no external library) |
| Observability | Langfuse (traces, scores, dashboards) |
| MCP Server | Model Context Protocol (Claude Desktop integration) |
| Containerization | Docker |
| Cloud Deployment | GCP Cloud Run + GKE + Serverless VPC Connector + Artifact Registry + Secret Manager |
| CI/CD | Cloud Build (GCP) |
| PDF Processing | PyMuPDF + Tesseract OCR |

---

## Project Structure

```
production-rag-langgraph/
├── README.md
├── cloudbuild.yaml                     # Cloud Build CI/CD pipeline
├── cloudrun-service.yaml               # Cloud Run service definition
├── gcp_setup.sh                        # One-time GCP setup script
├── render.yaml                         # Render deployment (secondary)
└── issue_support/
    ├── api.py                          # FastAPI REST API
    ├── app.py                          # Streamlit chat UI
    ├── mcp_server.py                   # MCP server (Claude Desktop)
    ├── Dockerfile
    ├── requirements.txt
    ├── .env.example
    ├── config/
    │   └── env_setup.py
    ├── data/
    │   ├── PC_trouble-shooting.pdf.dvc  # DVC pointer (committed to git)
    │   └── uploads/                     # Runtime upload staging area
    ├── graph/
    │   ├── workflow.py                  # LangGraph pipeline definition
    │   └── nodes/
    │       ├── guardrails_node.py       # 4-layer safety checks
    │       ├── router_node.py           # Score-gated routing + re-ranking
    │       ├── rag_node.py              # RAG generation with context
    │       ├── llm_node.py              # Generic LLM generation
    │       └── evaluator_llm_node.py    # LLM-as-Judge evaluation
    ├── memory/
    │   ├── state.py                     # LangGraph typed state schema
    │   ├── vector_store.py              # Milvus HNSW vector store
    │   └── BM25_keyword_search.py       # BM25 retriever factory
    ├── prompts/                         # System prompts for each node
    ├── tools/
    │   ├── document_loader.py           # PDF ingestion, chunking, GCS sync
    │   ├── rag_hybrid_retriever.py      # Hybrid search pipeline
    │   ├── rag_score.py                 # Retrieval scoring
    │   ├── ensemble_retriever_with_scores.py  # Custom RRF retriever
    │   ├── reranker.py                  # Cross-encoder re-ranker
    │   ├── evaluator_llm.py             # Evaluator chain
    │   └── llm_respond.py              # LLM response chain
    ├── utils/
    │   ├── helpers.py                   # LLM init, formatters, embeddings
    │   ├── gcs_store.py                 # GCS document persistence
    │   ├── metrics.py                   # JSONL metrics logging
    │   └── langfuse.py                  # Langfuse client
    ├── scripts/
    │   ├── evaluate.py                  # Evaluation harness
    │   ├── eval_results.json            # Baseline results (DVC-tracked)
    │   └── eval_results_v2.json         # Post-tuning results (DVC-tracked)
    └── tests/
        ├── test_api.py
        ├── test_helpers.py
        ├── test_metrics.py
        ├── test_reranker.py
        └── test_state.py
```

---

## License

MIT
