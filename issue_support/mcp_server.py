"""
MCP server for the IT Support RAG pipeline.

Exposes two tools:
  - query_it_support   : Ask a PC troubleshooting question
  - get_pipeline_metrics : Fetch aggregated performance stats

Run via stdio (Claude Desktop / any MCP client):
    python mcp_server.py

The server calls the FastAPI backend at API_URL (default http://localhost:8000).
Start the API first with:  docker-compose up  or  uvicorn api:app --port 8000
"""

import os
import httpx
from mcp.server.fastmcp import FastMCP

API_URL = os.getenv("API_URL", "http://localhost:8000")

mcp = FastMCP(
    "IT Support RAG",
    instructions=(
        "Use query_it_support for any PC or Windows troubleshooting question. "
        "The tool searches a curated knowledge base and returns a sourced answer. "
        "Use get_pipeline_metrics to inspect pipeline health."
    ),
)


@mcp.tool()
def query_it_support(query: str) -> str:
    """
    Query the IT support knowledge base for PC troubleshooting help.

    Sends the query through a hybrid RAG pipeline (BM25 + vector search,
    cross-encoder re-ranking, LLM-as-judge evaluation) and returns a
    sourced answer.

    Args:
        query: The user's PC or IT support question.

    Returns:
        Formatted answer with route taken, eval score, latency, and sources.
    """
    try:
        response = httpx.post(
            f"{API_URL}/chat",
            json={"query": query},
            timeout=60.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        return f"[Error contacting RAG API: {e}]"

    data = response.json()

    if data.get("guardrail_triggered"):
        return f"[BLOCKED — {data.get('guardrail_reason', 'policy')}]\n{data['answer']}"

    lines = [
        data["answer"],
        "",
        f"Route: {data['route']}  |  "
        f"Eval: {data['eval_score'] if data.get('eval_score') is not None else 'N/A'}/10  |  "
        f"Latency: {data['latency_ms']}ms",
    ]

    sources = data.get("sources") or []
    if sources:
        lines.append(f"\nSources ({len(sources)}):")
        for i, src in enumerate(sources, 1):
            parts = []
            if src.get("page") is not None:
                parts.append(f"page {src['page']}")
            if src.get("score") is not None:
                parts.append(f"reranker score {src['score']:.3f}")
            meta = " · ".join(parts)
            header = f"  {i}. [{meta}]" if meta else f"  {i}."
            lines.append(f"{header} {src['content'][:300].strip()}...")

    return "\n".join(lines)


@mcp.tool()
def get_pipeline_metrics() -> str:
    """
    Return aggregated performance metrics for the IT support RAG pipeline.

    Reports total request count, average latency, route distribution,
    average retrieval score, and average LLM-as-judge eval score.
    """
    try:
        response = httpx.get(f"{API_URL}/metrics", timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError as e:
        return f"[Error fetching metrics: {e}]"

    d = response.json()
    avg_rag = d.get("avg_rag_score")
    avg_eval = d.get("avg_eval_score")

    return "\n".join([
        "IT Support RAG — Pipeline Metrics",
        f"  Total requests  : {d['total_requests']}",
        f"  Avg latency     : {d['avg_latency_ms']} ms",
        f"  RAG route       : {d['rag_route_count']}",
        f"  LLM route       : {d['llm_route_count']}",
        f"  Avg RAG score   : {avg_rag if avg_rag is not None else 'N/A'}",
        f"  Avg eval score  : {avg_eval if avg_eval is not None else 'N/A'} / 10",
    ])


if __name__ == "__main__":
    mcp.run()
