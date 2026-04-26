# Production RAG Pipeline — Agentic IT Support Assistant

> Production-grade agentic RAG pipeline built with **LangGraph**, **hybrid BM25 + vector search**, **cross-encoder re-ranking**, **multi-layer guardrails**, and **LLM-as-judge evaluation** — served via **FastAPI**, Streamlit UI, **MCP server**, and containerized with **Docker Compose**.

[![CI Pipeline](https://github.com/NandaKumar8776/production-rag-langgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/NandaKumar8776/production-rag-langgraph/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple.svg)](https://github.com/langchain-ai/langgraph)
[![MCP](https://img.shields.io/badge/MCP-1.0-orange.svg)](https://modelcontextprotocol.io)

---

## Architecture

```
User Query
    |
    v
[FastAPI /chat endpoint]
    |
    v
+---------------------+
|   Guardrails Node   |  <-- Prompt injection, jailbreak, PII, abuse (4-layer check)
+--------+------------+
         |
    safe / blocked?
         |
    v (safe)      v (blocked) --> Refusal message
+-------------------+
|   Router Node     |  <-- Hybrid BM25 + Milvus HNSW retrieval
|   (Score-gated)   |      + Cross-encoder re-ranking (ms-marco-MiniLM)
+--------+----------+
         |
    score >= threshold?
    /              \
   v                v
[RAG Node]     [LLM Node]
(Groq LLM +    (Groq LLM,
 context)       general)
   \                /
    v              v
  [Evaluator Node]       <-- LLM-as-Judge (4-dimension rubric, 0-10)
         |
    eval_score + answer
         |
         v
   JSON Response + Streamlit UI + Langfuse trace + metrics.jsonl
```

## Key Technical Features

| Feature | Implementation |
|---------|---------------|
| **Multi-Layer Guardrails** | 4-stage safety check: prompt injection → jailbreak → PII → LLM abuse classifier |
| **Hybrid Retrieval** | BM25 (sparse) + Milvus HNSW (dense) with Reciprocal Rank Fusion |
| **Two-Stage Re-ranking** | Ensemble retrieval → cross-encoder re-ranker (`ms-marco-MiniLM-L-6-v2`) |
| **Score-Gated Routing** | Queries routed to RAG only when top retrieval score exceeds threshold |
| **LLM-as-Judge Evaluation** | 4-dimension rubric scoring (relevance, safety, actionability, completeness) |
| **Agentic Graph** | LangGraph StateGraph with conditional routing, typed state, Pydantic I/O validation |
| **Streamlit UI** | Chat interface with route badges, eval scores, reranker scores, source cards |
| **Dynamic PDF Ingestion** | `POST /ingest` — upload any PDF at runtime, indexed into Milvus + BM25 without restart |
| **Multi-Modal PDF Ingestion** | PyMuPDF with Tesseract OCR + markdown table extraction |
| **Observability** | Langfuse traces every node with latency, token counts, and eval scores |
| **Structured Metrics** | Per-request JSONL logging + `/metrics` API endpoint for aggregations |
| **MCP Server** | Exposes `query_it_support` and `get_pipeline_metrics` as tools for Claude Desktop |
| **Data Versioning** | DVC tracks the knowledge base PDF and eval artifacts — `dvc pull` to reproduce any version |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Groq API key ([get one here](https://console.groq.com))

### Run with Docker Compose

```bash
# 1. Clone the repository
git clone https://github.com/NandaKumar8776/production-rag-langgraph.git
cd production-rag-langgraph/issue_support

# 2. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Start all services (API + Milvus + etcd + MinIO)
docker-compose up --build

# 4. Test the API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "My PC wont boot after a Windows update"}'

# 5. Launch the Streamlit UI
streamlit run app.py

# 6. (Optional) Run the MCP server for Claude Desktop integration
python mcp_server.py
```

### Run Locally (Development)

```bash
cd issue_support

# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Milvus (requires Docker)
docker-compose up milvus-standalone -d

# 3. Run the API
uvicorn api:app --reload --port 8000

# 4. Run the UI (separate terminal)
streamlit run app.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a query, get RAG-powered response with metadata |
| `POST` | `/ingest` | Upload a PDF and add it to the knowledge base (no restart needed) |
| `GET` | `/health` | Liveness probe for container orchestration |
| `GET` | `/metrics` | Aggregated pipeline performance metrics |

### Example Response

```json
{
  "answer": "To fix a PC that won't boot after a Windows update...",
  "session_id": "abc-123",
  "route": "RAG",
  "top_rag_score": 0.87,
  "num_sources": 3,
  "sources": [
    {"content": "If Windows fails to boot...", "page": 5, "score": 5.29}
  ],
  "latency_ms": 1850.5,
  "eval_score": 8.2,
  "guardrail_triggered": false,
  "guardrail_reason": null
}
```

### Guardrail Block Example

```json
{
  "answer": "I'm not able to process that request...",
  "route": "BLOCKED",
  "guardrail_triggered": true,
  "guardrail_reason": "prompt_injection"
}
```

## Live Demo Deployment

The API can be deployed to [Render](https://render.com) (free tier) using the included `render.yaml`. The vector store supports three modes so you can pick the one that fits your deployment:

| Mode | `MILVUS_URI` value | When to use |
|---|---|---|
| **Milvus Lite** | `./milvus_demo.db` | Quickest deploy — no external service, data resets on redeploy |
| **Zilliz Cloud** | `https://in03-xxx...zillizcloud.com` | Persistent demo — free tier at [cloud.zilliz.com](https://cloud.zilliz.com) |
| **Local Docker** | `http://localhost:19530` | Local development with full Docker Compose |

### Deploy to Render (Milvus Lite — fastest path)

```bash
# 1. Fork the repo and connect it to Render
# 2. Render auto-detects render.yaml — click "Apply"
# 3. Set GROQ_API_KEY in the Render dashboard under Environment
# 4. Deploy — the API will be live at https://it-support-rag.onrender.com
```

### Deploy to Render (Zilliz Cloud — persistent)

```bash
# 1. Create a free cluster at https://cloud.zilliz.com
# 2. Copy the Public Endpoint and API Key
# 3. In render.yaml, comment out the Lite option and uncomment the Zilliz block
# 4. Set GROQ_API_KEY and ZILLIZ_API_KEY in the Render dashboard
# 5. Deploy
```

> **Note:** Render free tier spins down after 15 minutes of inactivity — expect a ~30s cold start on the first request.

---

## Data Versioning (DVC)

Large files are tracked with [DVC](https://dvc.org) instead of git. Git stores the full content of every committed file forever — committing a large PDF bloats the repo history permanently and hits GitHub's 100 MB file limit. DVC stores the actual file in a separate cache and commits only a tiny `.dvc` pointer (a hash + path, a few lines of text). Running `dvc pull` after a clone fetches the real files.

### Tracked Artifacts

| File | Why DVC |
|---|---|
| `data/PC_trouble-shooting.pdf` | Knowledge base — swapping in a larger corpus later won't balloon git history |
| `scripts/eval_results.json` | Baseline eval snapshot — versioned alongside the code that produced it |
| `scripts/eval_results_v2.json` | Post-tuning eval snapshot — reproducible by checking out the matching git tag |

### Usage

```bash
# After cloning — restore all tracked files
dvc pull

# Add a new knowledge base document
dvc add data/new-corpus.pdf
dvc push

# Re-run evaluation and version the new results
python scripts/evaluate.py --output scripts/eval_results_v3.json
dvc add scripts/eval_results_v3.json
dvc push

# Switch from the default local cache to a cloud remote (S3, GCS, Azure)
dvc remote add -d myremote s3://your-bucket/dvc-cache
dvc push
```

---

## MCP Server

The pipeline is also served as an [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server, letting Claude Desktop — or any MCP-compatible client — call the RAG pipeline as a native tool.

### Tools Exposed

| Tool | Description |
|------|-------------|
| `query_it_support` | Ask a PC troubleshooting question — runs the full RAG pipeline and returns a sourced answer |
| `get_pipeline_metrics` | Fetch aggregated latency, route distribution, and eval scores |

### Setup

```bash
# 1. Install MCP dependency
pip install "mcp[cli]>=1.0.0"

# 2. Start the API (must be running first)
docker-compose up -d

# 3. Run the MCP server
cd issue_support
python mcp_server.py
```

### Claude Desktop Integration

Add to your Claude Desktop `claude_desktop_config.json` (see `issue_support/claude_desktop_config.json` for a template):

```json
{
  "mcpServers": {
    "it-support-rag": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/absolute/path/to/issue_support",
      "env": { "API_URL": "http://localhost:8000" }
    }
  }
}
```

Once connected, Claude can call `query_it_support` and `get_pipeline_metrics` directly in conversation.

---

## Pipeline Performance

Evaluated against a 12-query golden test set (8 PC troubleshooting + 4 off-topic) using `python scripts/evaluate.py`.

| Metric | Value |
|--------|-------|
| Avg eval score (rubric, 0-10) | **7.88** |
| RAG routing precision | **100%** |
| Overall routing precision | **66.7%** |
| Avg end-to-end latency | **3,094ms** |
| Chunk recall@3 | **100%** |

> Re-run evaluation: `python scripts/evaluate.py --output scripts/eval_results.json`

## Tech Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph (StateGraph, conditional edges) |
| Vector Database | Milvus (HNSW index, L2 metric) |
| Sparse Retrieval | BM25 (rank-bm25) |
| Re-ranking | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| LLM Inference | Groq (Llama-3.3-70b, Llama-4-Scout) |
| API Framework | FastAPI + Pydantic v2 |
| Frontend | Streamlit |
| Data Versioning | DVC (tracks knowledge base PDFs and eval artifacts outside git) |
| MCP Server | Model Context Protocol (Claude Desktop tool integration) |
| Guardrails | Regex + LLM classifier (no external library) |
| Observability | Langfuse (traces, scores, dashboards) |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions (test + lint + Docker build) |
| PDF Processing | PyMuPDF + Tesseract OCR |

## Project Structure

```
production-rag-langgraph/
+-- README.md
+-- render.yaml                         # Render deployment config (free tier)
+-- issue_support/                      # Main project directory
    +-- api.py                          # FastAPI REST API
    +-- app.py                          # Streamlit chat UI
    +-- main.py                         # CLI chatbot interface
    +-- mcp_server.py                   # MCP server (Claude Desktop integration)
    +-- claude_desktop_config.json      # Claude Desktop config template
    +-- Dockerfile                      # Container image
    +-- docker-compose.yml              # Full-stack deployment
    +-- requirements.txt                # Python dependencies
    +-- .env.example                    # Environment variable template
    +-- config/
    |   +-- env_setup.py                # Environment configuration
    +-- data/
    |   +-- PC_trouble-shooting.pdf     # Knowledge base document (DVC-tracked)
    |   +-- PC_trouble-shooting.pdf.dvc # DVC pointer (committed to git)
    +-- graph/
    |   +-- workflow.py                 # LangGraph pipeline definition
    |   +-- nodes/
    |       +-- guardrails_node.py      # 4-layer safety checks
    |       +-- router_node.py          # Score-gated routing + re-ranking
    |       +-- rag_node.py             # RAG generation with context
    |       +-- llm_node.py             # Generic LLM generation
    |       +-- evaluator_llm_node.py   # LLM-as-Judge evaluation
    |       +-- get_details_node.py     # Detail extraction node
    +-- memory/
    |   +-- state.py                    # LangGraph typed state schema
    |   +-- vector_store.py             # Milvus HNSW vector store
    |   +-- BM25_keyword_search.py      # BM25 retriever factory
    +-- prompts/
    |   +-- rag_prompt.txt              # RAG generation prompt
    |   +-- llm_prompt.txt              # Generic LLM prompt
    |   +-- router_prompt.txt           # Router classification prompt
    |   +-- evaluator_llm_prompt.txt    # LLM-as-Judge rubric prompt
    |   +-- guardrails_llm_prompt.txt   # Guardrails abuse classifier prompt
    +-- tools/
    |   +-- document_loader.py          # PDF ingestion + chunking
    |   +-- rag_hybrid_retriever.py     # Hybrid search pipeline
    |   +-- rag_score.py                # Retrieval scoring
    |   +-- ensemble_retriever_with_scores.py  # Custom RRF retriever
    |   +-- reranker.py                 # Cross-encoder re-ranker
    |   +-- evaluator_llm.py            # Evaluator chain definition
    |   +-- llm_respond.py              # LLM response chain
    +-- utils/
    |   +-- helpers.py                  # LLM init, formatters, embeddings
    |   +-- metrics.py                  # Structured JSONL metrics logging
    |   +-- langfuse.py                 # Langfuse client setup
    +-- scripts/
    |   +-- evaluate.py                 # Evaluation harness (golden test set)
    |   +-- eval_results.json           # Baseline evaluation results (DVC-tracked)
    |   +-- eval_results_v2.json        # Post-tuning evaluation results (DVC-tracked)
    +-- tests/                          # pytest test suite
    |   +-- test_api.py
    |   +-- test_helpers.py
    |   +-- test_metrics.py
    |   +-- test_reranker.py
    |   +-- test_state.py
    +-- .github/workflows/ci.yml        # GitHub Actions CI pipeline
```

## Running Tests

```bash
cd issue_support
pytest tests/ -v
```

## License

MIT
