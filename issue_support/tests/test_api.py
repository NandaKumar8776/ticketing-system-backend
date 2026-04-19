"""
Tests for the FastAPI REST API layer.

These tests use httpx's AsyncClient to test endpoints without starting
the actual LangGraph pipeline (which requires Milvus, Groq API, etc.).
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client that bypasses lifespan initialization."""
    # Patch the heavy startup dependencies
    with patch("config.env_setup", create=True), \
         patch("tools.document_loader.initialize_retrievers"):
        from api import app
        return TestClient(app, raise_server_exceptions=False)


def test_health_endpoint(client):
    """GET /health should return 200 with status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_metrics_endpoint_empty(client):
    """GET /metrics should return zeros when no requests have been made."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] == 0
    assert data["avg_latency_ms"] == 0.0


def test_chat_request_validation(client):
    """POST /chat should reject empty queries."""
    response = client.post("/chat", json={"query": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_chat_request_model():
    """ChatRequest model should auto-generate session_id if not provided."""
    from api import ChatRequest

    req = ChatRequest(query="test question")
    assert req.query == "test question"
    assert req.session_id is not None
    assert len(req.session_id) > 0


def test_chat_response_model():
    """ChatResponse model should accept all expected fields."""
    from api import ChatResponse

    resp = ChatResponse(
        answer="Test answer",
        session_id="abc-123",
        route="RAG",
        top_rag_score=0.85,
        num_sources=3,
        sources=[],
        latency_ms=150.5,
        eval_score=8.5,
    )
    assert resp.route == "RAG"
    assert resp.eval_score == 8.5
