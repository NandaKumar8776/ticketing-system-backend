from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from operator import add

# Defining the state memory for LangGraph

class State(TypedDict):
    """
    State memory structure for the Issue Support workflow.

    This TypedDict defines the structure of the state passed between LangGraph nodes.
    Messages are managed using the add_messages reducer, which automatically handles
    message merging. Additional fields are stored separately to avoid polluting
    messages with additional_kwargs.

    Fields:
        messages: Conversation history managed by LangGraph's add_messages reducer.
        current_question: Latest user question (appended via add operator).
        context: Retrieved context documents for RAG (set by router_node).
        category: Routing decision — "RAG relevant Issue" or "Not Related".
        top_rag_score: Highest retrieval score from hybrid search (for metrics).
        eval_score: LLM-as-judge evaluation score 0-10 (set by evaluator_node).
    """
    messages: Annotated[list, add_messages]
    current_question: Annotated[list[str], add]
    context: list                        # Context documents for RAG
    category: str                        # Routing category
    top_rag_score: Optional[float]       # Best retrieval score (metrics)
    eval_score: Optional[float]          # Evaluator rubric score 0-10
    guardrail_triggered: bool            # True if request was blocked by guardrails
    guardrail_reason: Optional[str]      # Reason for block: "prompt_injection", "jailbreak", "pii", "off_topic_abuse"


    