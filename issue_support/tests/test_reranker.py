"""
Tests for the cross-encoder re-ranker.
"""

import pytest
from langchain_core.documents import Document


def test_rerank_returns_top_n(mock_documents, sample_question):
    """Re-ranker should return exactly top_n documents sorted by relevance."""
    from tools.reranker import rerank_documents

    results = rerank_documents(
        query=sample_question,
        documents=mock_documents,
        top_n=2,
    )

    assert len(results) == 2
    # Each result is a (Document, float) tuple
    for doc, score in results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)


def test_rerank_scores_are_descending(mock_documents, sample_question):
    """Re-ranked results should be sorted by score in descending order."""
    from tools.reranker import rerank_documents

    results = rerank_documents(
        query=sample_question,
        documents=mock_documents,
        top_n=4,
    )

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_relevant_docs_score_higher(mock_documents, sample_question):
    """Screen-related docs should score higher than unrelated network doc."""
    from tools.reranker import rerank_documents

    results = rerank_documents(
        query=sample_question,
        documents=mock_documents,
        top_n=4,
    )

    # The last document (network troubleshooting) should be ranked lowest
    last_doc_content = results[-1][0].page_content
    assert "network" in last_doc_content.lower() or "router" in last_doc_content.lower()


def test_rerank_empty_docs():
    """Re-ranker should handle empty document list gracefully."""
    from tools.reranker import rerank_documents

    results = rerank_documents(query="test query", documents=[], top_n=3)
    assert results == []
