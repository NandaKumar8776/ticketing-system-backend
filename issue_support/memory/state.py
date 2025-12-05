from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from operator import add

# Defining the state memory for LangGraph

class State(TypedDict):
    """
    State memory structure for the Issue Support workflow.

    Fields:
    messages (list): List of messages in the conversation history. Each message is a dict with 'role' and 'content'.
    current_question (list[str]): List containing the latest user question as a string. Will add to list when giving it a list of string(s).
    context (list): Optional list of context documents retrieved for RAG. Stored separately to avoid additional_kwargs.
    category (str): Optional routing category ("PC Issue" or "Not Related"). Stored separately to avoid additional_kwargs.

    """
    messages: Annotated[list, add_messages]
    current_question: Annotated[list[str], add]
    context: list  # Optional: context documents for RAG
    category: str  # Optional: routing category 


    