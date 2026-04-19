"""
Cross-Encoder Re-Ranker for two-stage retrieval.

Stage 1: Hybrid BM25 + Vector retrieval returns top-K candidates (handled by EnsembleRetriever).
Stage 2: This cross-encoder re-scores candidates using query-document attention and returns top-N.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (lightweight, ~22M params, fast inference).
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Singleton model instance (lazy loaded)
_reranker_model: Optional[CrossEncoder] = None

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker() -> CrossEncoder:
    """
    Get or initialize the cross-encoder re-ranker model (singleton).

    Returns:
        CrossEncoder: The initialized cross-encoder model.
    """
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"Loading cross-encoder model: {RERANKER_MODEL_NAME}")
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
        logger.info("Cross-encoder model loaded successfully.")
    return _reranker_model


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int = 3,
) -> list[tuple[Document, float]]:
    """
    Re-rank retrieved documents using cross-encoder attention scoring.

    The cross-encoder jointly encodes (query, document) pairs and produces a
    relevance score with full attention between query and document tokens.
    This is more accurate than bi-encoder similarity but slower — hence used
    as a second-stage re-ranker on a small candidate set.

    Args:
        query: The user's search query.
        documents: List of candidate Document objects from Stage 1 retrieval.
        top_n: Number of top-ranked documents to return.

    Returns:
        List of (Document, score) tuples, sorted by cross-encoder score descending.
    """
    if not documents:
        logger.warning("No documents provided for re-ranking.")
        return []

    model = get_reranker()

    # Build (query, doc_content) pairs for cross-encoder
    pairs = []
    for doc in documents:
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        pairs.append((query, content))

    # Score all pairs
    scores = model.predict(pairs)

    # Combine documents with their scores and sort descending
    scored_docs = list(zip(documents, scores.tolist()))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    logger.info(
        f"Re-ranked {len(documents)} docs → top {top_n}. "
        f"Score range: [{scored_docs[-1][1]:.4f}, {scored_docs[0][1]:.4f}]"
    )

    return scored_docs[:top_n]
