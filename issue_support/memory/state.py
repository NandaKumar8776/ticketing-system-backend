from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from operator import add

# Defining the state memory for LangGraph

class State(TypedDict):
    """
    State memory structure for the Issue Support workflow.
    
    This TypedDict defines the structure of the state passed between LangGraph nodes.
    Messages are managed using the add_messages reducer, which automatically handles
    message merging. Additional fields (context, category) are stored separately to
    avoid polluting messages with additional_kwargs.

    Fields:
        messages (Annotated[list, add_messages]): List of messages in the conversation history.
            Each message is a dict with 'role' and 'content'. Managed by LangGraph's add_messages reducer.
        current_question (Annotated[list[str], add]): List containing the latest user question as a string.
            Uses the 'add' operator to append new questions.
        context (list): Optional list of context documents retrieved for RAG.
            Set by router_node when routing to RAG. Stored in state field (not in message additional_kwargs).
        category (str): Optional routing category ("RAG relevant Issue" or "Not Related").
            Set by router_node to indicate routing decision. Stored in state field (not in message additional_kwargs).
    """
    messages: Annotated[list, add_messages]
    current_question: Annotated[list[str], add]
    context: list  # Optional: context documents for RAG
    category: str  # Optional: routing category 


    