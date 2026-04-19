"""
Tests for utility helper functions.
"""

import os
import tempfile
import pytest


def test_read_prompt():
    """read_prompt should read a text file and return its content."""
    from utils.helpers import read_prompt

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a test prompt with {variable}")
        f.flush()
        content = read_prompt(f.name)

    assert content == "This is a test prompt with {variable}"
    os.unlink(f.name)


def test_output_formatter():
    """output_formatter should extract content from the last message."""
    from utils.helpers import output_formatter
    from unittest.mock import MagicMock

    mock_msg = MagicMock()
    mock_msg.content = "Test response"

    state = {"messages": [mock_msg]}
    result = output_formatter(state)
    assert result == "Test response"


def test_doc_output_formatter(mock_documents):
    """doc_output_formatter should join document contents with double newlines."""
    from utils.helpers import doc_output_formatter

    result = doc_output_formatter(mock_documents)
    assert "screen is flickering" in result
    assert "cable connections" in result
    # Documents should be separated by double newlines
    assert "\n\n" in result


def test_doc_output_formatter_empty():
    """doc_output_formatter should handle empty list."""
    from utils.helpers import doc_output_formatter

    result = doc_output_formatter([])
    assert result == ""
