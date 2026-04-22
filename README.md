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
    +-- app.py                          # Streamlit UI (legacy)
    +-- Dockerfile                      # Container image
    +-- docker-compose.yml              # Full-stack deployment
    +-- requirements.txt                # Python dependencies
    +-- .env.example                    # Environment variable template
    +-- config/
    |   +-- env_setup.py                # Environment configuration
    +-- data/
    |   +-- PC_trouble-shooting.pdf     # Knowledge base document
    +-- graph/
    |   +-- workflow.py                 # LangGraph pipeline definition
    |   +-- nodes/
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
    |   +-- eval_results.json           # Baseline evaluation results
    |   +-- eval_results_v2.json        # Post-tuning evaluation results
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
