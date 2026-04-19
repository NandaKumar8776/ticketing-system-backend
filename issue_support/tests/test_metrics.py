"""
Tests for the metrics logging module.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch


def test_log_metrics_writes_jsonl():
    """log_metrics should append a JSON line to the metrics file."""
    from utils.metrics import log_metrics, METRICS_FILE

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        temp_path = Path(f.name)

    with patch("utils.metrics.METRICS_FILE", temp_path):
        log_metrics({"route": "RAG", "latency_ms": 120.5})
        log_metrics({"route": "LLM", "latency_ms": 80.3})

    lines = temp_path.read_text().strip().split("\n")
    assert len(lines) == 2

    record = json.loads(lines[0])
    assert record["route"] == "RAG"
    assert record["latency_ms"] == 120.5
    assert "timestamp" in record

    temp_path.unlink()


def test_read_metrics():
    """read_metrics should return recent records from the metrics file."""
    from utils.metrics import log_metrics, read_metrics, METRICS_FILE

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        temp_path = Path(f.name)

    with patch("utils.metrics.METRICS_FILE", temp_path):
        for i in range(5):
            log_metrics({"request_id": i, "route": "RAG"})

        records = read_metrics(last_n=3)

    assert len(records) == 3
    assert records[0]["request_id"] == 2  # last 3: indices 2, 3, 4

    temp_path.unlink()
