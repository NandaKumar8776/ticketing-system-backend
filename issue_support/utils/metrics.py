"""
Structured metrics logging for the RAG pipeline.

Logs per-request metrics to a JSONL file for offline analysis, and optionally
pushes scores to Langfuse for dashboard visualization.

Metrics logged:
- session_id, route, top_rag_score, num_sources, latency_ms, eval_score,
  query_length, timestamp
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

METRICS_FILE = Path(__file__).parent.parent / "metrics.jsonl"


def log_metrics(record: dict) -> None:
    """
    Append a metrics record to the JSONL log file.

    Args:
        record: Dictionary of metric key-value pairs.
    """
    if "timestamp" not in record:
        record["timestamp"] = time.time()

    try:
        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        logger.error(f"Failed to write metrics: {e}")


def log_to_langfuse(
    trace_id: str,
    eval_score: Optional[float] = None,
    latency_ms: Optional[float] = None,
    route: Optional[str] = None,
) -> None:
    """
    Push evaluation scores to Langfuse for dashboard tracking.

    Args:
        trace_id: The Langfuse trace ID to attach scores to.
        eval_score: LLM-as-judge overall score (0-10).
        latency_ms: End-to-end latency in milliseconds.
        route: Pipeline route taken ("RAG" or "LLM").
    """
    try:
        from langfuse import get_client
        langfuse = get_client()

        if eval_score is not None:
            langfuse.score(
                trace_id=trace_id,
                name="eval_overall",
                value=eval_score,
                comment=f"Route: {route}, Latency: {latency_ms}ms",
            )

        if latency_ms is not None:
            langfuse.score(
                trace_id=trace_id,
                name="latency_ms",
                value=latency_ms,
            )

        logger.info(f"Langfuse scores pushed for trace {trace_id}")
    except Exception as e:
        logger.warning(f"Failed to push scores to Langfuse: {e}")


def read_metrics(last_n: int = 100) -> list[dict]:
    """
    Read the most recent N metrics records from the JSONL log.

    Args:
        last_n: Number of recent records to return.

    Returns:
        List of metric dictionaries, most recent first.
    """
    if not METRICS_FILE.exists():
        return []

    records = []
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to read metrics: {e}")

    return records[-last_n:]
