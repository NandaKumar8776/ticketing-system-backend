"""
Shared pytest fixtures for the Issue Support RAG test suite.
"""

import os
import sys
import pytest

# Ensure the project root is on sys.path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def sample_question():
    """A typical PC troubleshooting question for testing."""
    return "My laptop screen is flickering. How can I fix it?"


@pytest.fixture
def off_topic_question():
    """A question unrelated to PC troubleshooting."""
    return "What is the capital of France?"


@pytest.fixture
def mock_documents():
    """Mock LangChain Document objects for retriever tests."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="If your screen is flickering, try updating your display drivers.",
            metadata={"source": "test.pdf", "page": 1},
        ),
        Document(
            page_content="Check your screen cable connections and ensure they are secure.",
            metadata={"source": "test.pdf", "page": 2},
        ),
        Document(
            page_content="Adjust the refresh rate in Display Settings to match your monitor.",
            metadata={"source": "test.pdf", "page": 3},
        ),
        Document(
            page_content="Unrelated content about network troubleshooting for routers.",
            metadata={"source": "test.pdf", "page": 10},
        ),
    ]
