# Agentic RAG System - PC Troubleshooting Assistant

> Production-grade agentic RAG pipeline built with **LangGraph**, **hybrid BM25 + vector search**, **cross-encoder re-ranking**, and **LLM-as-judge evaluation** - served via **FastAPI** and containerized with **Docker Compose**.

[![CI Pipeline](https://github.com/NandaKumar8776/ticketing-system-backend/actions/workflows/ci.yml/badge.svg)](https://github.com/NandaKumar8776/ticketing-system-backend/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple.svg)](https://github.com/langchain-ai/langgraph)

---

## Architecture

```
User Query
    |
    v
[FastAPI /chat endpoint]
    |
    v
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
   JSON Response + Langfuse trace + metrics.jsonl
```

## Key Technical Features

| Feature | Implementation |
|---------|---------------|
| **Hybrid Retrieval** | BM25 (sparse) + Milvus HNSW (dense) with Reciprocal Rank Fusion |
| **Two-Stage Re-ranking** | Ensemble retrieval -> cross-encoder re-ranker (`ms-marco-MiniLM-L-6-v2`) |
| **Score-Gated Routing** | Queries routed to RAG only when top retrieval score exceeds threshold |
| **LLM-as-Judge Evaluation** | 4-dimension rubric scoring (relevance, safety, actionability, completeness) |
| **Agentic Graph** | LangGraph StateGraph with conditional routing, typed state, Pydantic I/O validation |
| **Multi-Modal PDF Ingestion** | PyMuPDF with Tesseract OCR + markdown table extraction |
| **Observability** | Langfuse traces every node with latency, token counts, and eval scores |
| **Structured Metrics** | Per-request JSONL logging + `/metrics` API endpoint for aggregations |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Groq API key ([get one here](https://console.groq.com))

### Run with Docker Compose

```bash
# 1. Clone the repository
git clone https://github.com/NandaKumar8776/ticketing-system-backend.git
cd ticketing-system-backend/issue_support

# 2. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Start all services (API + Milvus + etcd + MinIO)
docker-compose up --build

# 4. Test the API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "My PC wont boot after a Windows update"}'
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
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a query, get RAG-powered response with metadata |
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
    {"content": "If Windows fails to boot...", "page": 5, "score": 0.92}
  ],
  "latency_ms": 1850.5,
  "eval_score": 8.2
}
```

## Pipeline Performance

| Metric | Value |
|--------|-------|
| Avg eval score (rubric, 0-10) | TBD |
| RAG routing precision | TBD |
| Avg retrieval latency (hybrid + re-rank) | TBD |
| Avg end-to-end latency | TBD |
| Chunk recall@3 | TBD |

> You can test these metrics yourself by running `python scripts/evaluate.py` against a golden test set.

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
| Observability | Langfuse (traces, scores, dashboards) |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions (test + lint + Docker build) |
| PDF Processing | PyMuPDF + Tesseract OCR |

## Project Structure

```
ticketing-system-backend/
+-- README.md
+-- issue_support/                      # Main project directory
    +-- api.py                          # FastAPI REST API
    +-- main.py                         # CLI chatbot interface
    +-- Dockerfile                      # Container image
    +-- docker-compose.yml              # Full-stack deployment
    +-- requirements.txt                # Python dependencies
    +-- .env.example                    # Environment variable template
    +-- config/
    |   +-- env_setup.py                # Environment configuration
    +-- graph/
    |   +-- workflow.py                 # LangGraph pipeline definition
    |   +-- nodes/
    |       +-- router_node.py          # Score-gated routing + re-ranking
    |       +-- rag_node.py             # RAG generation with context
    |       +-- llm_node.py             # Generic LLM generation
    |       +-- evaluator_llm_node.py   # LLM-as-Judge evaluation
    +-- tools/
    |   +-- reranker.py                 # Cross-encoder re-ranker
    |   +-- rag_hybrid_retriever.py     # Hybrid search pipeline
    |   +-- rag_score.py                # Retrieval scoring
    |   +-- ensemble_retriever_with_scores.py  # Custom RRF retriever
    |   +-- document_loader.py          # PDF ingestion + chunking
    |   +-- evaluator_llm.py            # Evaluator chain definition
    |   +-- llm_respond.py              # LLM response chain
    +-- memory/
    |   +-- state.py                    # LangGraph typed state schema
    |   +-- vector_store.py             # Milvus HNSW vector store
    |   +-- BM25_keyword_search.py      # BM25 retriever factory
    +-- prompts/                        # Prompt templates (RAG, LLM, evaluator)
    +-- utils/
    |   +-- helpers.py                  # LLM init, formatters, embeddings
    |   +-- metrics.py                  # Structured JSONL metrics logging
    |   +-- langfuse.py                 # Langfuse client setup
    +-- tests/                          # pytest test suite
    +-- .github/workflows/ci.yml        # GitHub Actions CI pipeline
```

## Running Tests

```bash
cd issue_support
pytest tests/ -v
```

## License

MIT
