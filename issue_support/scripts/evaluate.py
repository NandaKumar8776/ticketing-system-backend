"""
Evaluation harness for the Issue Support RAG pipeline.

Runs a golden test set against the live API at localhost:8000 and reports:
  - Avg eval score (LLM-as-Judge rubric, 0-10)
  - RAG routing precision (% of PC queries correctly routed to RAG)
  - Avg retrieval latency (approximated from total latency split)
  - Avg end-to-end latency
  - Chunk recall@3 (% of queries where expected keywords appear in top-3 sources)

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --api-url http://localhost:8000 --output results.json
"""

import argparse
import json
import sys
import time
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Golden test set
# Each entry:
#   query           : the user query sent to POST /chat
#   expected_route  : "RAG" (PC-related) or "LLM" (off-topic)
#   expected_keywords: list of strings that should appear in the top-3 retrieved
#                      source chunks for RAG queries (used for recall@3)
# ---------------------------------------------------------------------------
GOLDEN_SET = [
    # --- PC troubleshooting queries (should route to RAG) ---
    {
        "query": "My PC won't turn on or start. What should I check?",
        "expected_route": "RAG",
        "expected_keywords": ["power", "cable", "reseat", "button"],
    },
    {
        "query": "I see an 'Invalid system disk' error when booting. How do I fix it?",
        "expected_route": "RAG",
        "expected_keywords": ["disk", "spacebar", "remove", "start"],
    },
    {
        "query": "The PC does not respond when I press the power button.",
        "expected_route": "RAG",
        "expected_keywords": ["hold", "button", "turn off", "on button"],
    },
    {
        "query": "My screen is blank after turning on the computer.",
        "expected_route": "RAG",
        "expected_keywords": ["monitor", "display", "cable", "brightness", "blank"],
    },
    {
        "query": "The computer is running very slowly. How do I speed it up?",
        "expected_route": "RAG",
        "expected_keywords": ["memory", "disk", "program", "startup", "slow"],
    },
    {
        "query": "My keyboard is not working. What should I do?",
        "expected_route": "RAG",
        "expected_keywords": ["keyboard", "usb", "port", "restart", "driver"],
    },
    {
        "query": "The PC keeps restarting on its own randomly.",
        "expected_route": "RAG",
        "expected_keywords": ["restart", "overheat", "driver", "update", "crash"],
    },
    {
        "query": "How do I recover my files after a hard drive failure?",
        "expected_route": "RAG",
        "expected_keywords": ["backup", "drive", "recovery", "data"],
    },
    # --- Off-topic queries (should route to LLM, not RAG) ---
    {
        "query": "What is the capital of France?",
        "expected_route": "LLM",
        "expected_keywords": [],
    },
    {
        "query": "Can you write me a poem about autumn?",
        "expected_route": "LLM",
        "expected_keywords": [],
    },
    {
        "query": "What is the best recipe for chocolate cake?",
        "expected_route": "LLM",
        "expected_keywords": [],
    },
    {
        "query": "Tell me a joke.",
        "expected_route": "LLM",
        "expected_keywords": [],
    },
]


def send_query(api_url: str, query: str, session_id: Optional[str] = None) -> dict:
    """Send a single query to POST /chat and return the parsed response."""
    payload = {"query": query}
    if session_id:
        payload["session_id"] = session_id

    response = httpx.post(
        f"{api_url}/chat",
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def check_recall_at_3(sources: list[dict], keywords: list[str]) -> bool:
    """
    Return True if any expected keyword appears in the top-3 source chunks.
    Case-insensitive substring match.
    """
    if not keywords:
        return True  # N/A for LLM-routed queries

    top3 = sources[:3]
    combined = " ".join(s.get("content", "") for s in top3).lower()
    return any(kw.lower() in combined for kw in keywords)


def run_evaluation(api_url: str) -> dict:
    results = []
    print(f"\nRunning evaluation against {api_url}")
    print(f"{'#':<4} {'Route':>6} {'Expected':>8} {'Eval':>6} {'Latency':>9} {'Recall':>7}  Query")
    print("-" * 90)

    for i, test in enumerate(GOLDEN_SET, 1):
        query = test["query"]
        expected_route = test["expected_route"]
        keywords = test["expected_keywords"]

        try:
            resp = send_query(api_url, query)

            route = resp.get("route", "UNKNOWN")
            eval_score = resp.get("eval_score")
            latency_ms = resp.get("latency_ms", 0.0)
            sources = resp.get("sources", [])
            recall = check_recall_at_3(sources, keywords)
            route_correct = route == expected_route

            results.append({
                "query": query,
                "expected_route": expected_route,
                "actual_route": route,
                "route_correct": route_correct,
                "eval_score": eval_score,
                "latency_ms": latency_ms,
                "num_sources": len(sources),
                "recall_at_3": recall if keywords else None,
            })

            recall_str = ("YES" if recall else "NO ") if keywords else "N/A"
            eval_str = f"{eval_score:.2f}" if eval_score is not None else " N/A"
            route_str = f"{'OK' if route_correct else 'XX'} {route}"

            print(
                f"{i:<4} {route_str:>7} {expected_route:>8} {eval_str:>6} "
                f"{latency_ms:>7.0f}ms {recall_str:>7}  {query[:55]}"
            )

        except Exception as e:
            print(f"{i:<4} ERROR: {e}  Query: {query[:55]}")
            results.append({
                "query": query,
                "expected_route": expected_route,
                "actual_route": "ERROR",
                "route_correct": False,
                "eval_score": None,
                "latency_ms": None,
                "num_sources": 0,
                "recall_at_3": None,
                "error": str(e),
            })

    return results


def compute_summary(results: list[dict]) -> dict:
    total = len(results)
    successful = [r for r in results if r.get("actual_route") != "ERROR"]

    # Routing precision: % of all queries where route matched expected
    routing_correct = sum(1 for r in successful if r["route_correct"])
    routing_precision = routing_correct / len(successful) * 100 if successful else 0.0

    # RAG routing precision: among PC queries only, % routed to RAG
    pc_queries = [r for r in successful if r["expected_route"] == "RAG"]
    rag_correct = sum(1 for r in pc_queries if r["actual_route"] == "RAG")
    rag_routing_precision = rag_correct / len(pc_queries) * 100 if pc_queries else 0.0

    # Avg eval score
    scored = [r["eval_score"] for r in successful if r["eval_score"] is not None]
    avg_eval = sum(scored) / len(scored) if scored else None

    # Avg latency
    latencies = [r["latency_ms"] for r in successful if r["latency_ms"] is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    # Recall@3 (PC queries only)
    recall_results = [r["recall_at_3"] for r in successful if r["recall_at_3"] is not None]
    recall_at_3 = sum(recall_results) / len(recall_results) * 100 if recall_results else None

    return {
        "total_queries": total,
        "successful": len(successful),
        "routing_precision_pct": round(routing_precision, 1),
        "rag_routing_precision_pct": round(rag_routing_precision, 1),
        "avg_eval_score": round(avg_eval, 2) if avg_eval is not None else None,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency is not None else None,
        "recall_at_3_pct": round(recall_at_3, 1) if recall_at_3 is not None else None,
    }


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("PIPELINE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total queries run         : {summary['total_queries']}")
    print(f"  Successful                : {summary['successful']}")
    print(f"  Avg eval score (0-10)     : {summary['avg_eval_score'] or 'N/A'}")
    print(f"  RAG routing precision     : {summary['rag_routing_precision_pct']}%")
    print(f"  Overall routing precision : {summary['routing_precision_pct']}%")
    print(f"  Avg end-to-end latency    : {summary['avg_latency_ms']}ms")
    print(f"  Chunk recall@3            : {summary['recall_at_3_pct']}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Issue Support RAG pipeline")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--output", default=None, help="Save full results to a JSON file")
    args = parser.parse_args()

    # Verify API is reachable
    try:
        health = httpx.get(f"{args.api_url}/health", timeout=10.0)
        health.raise_for_status()
    except Exception as e:
        print(f"ERROR: API not reachable at {args.api_url} — {e}")
        sys.exit(1)

    results = run_evaluation(args.api_url)
    summary = compute_summary(results)
    print_summary(summary)

    if args.output:
        output = {"summary": summary, "results": results, "timestamp": time.time()}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
