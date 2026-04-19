"""
Tests for the LangGraph State schema.
"""

from memory.state import State


def test_state_has_required_fields():
    """State TypedDict should define all required workflow fields."""
    annotations = State.__annotations__

    required_fields = [
        "messages",
        "current_question",
        "context",
        "category",
        "top_rag_score",
        "eval_score",
    ]

    for field in required_fields:
        assert field in annotations, f"State missing required field: {field}"


def test_state_instantiation():
    """State should be instantiable as a dict with all fields."""
    state: State = {
        "messages": [],
        "current_question": [],
        "context": [],
        "category": "Not Related",
        "top_rag_score": None,
        "eval_score": None,
    }
    assert state["category"] == "Not Related"
    assert state["eval_score"] is None
