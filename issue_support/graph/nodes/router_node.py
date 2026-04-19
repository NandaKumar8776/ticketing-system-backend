from memory.state import State
from pydantic import BaseModel, Field
from tools.rag_score import extract_docs_and_scores
from tools.reranker import rerank_documents
from langchain_core.messages import AIMessage
import os
import logging

logger = logging.getLogger(__name__)

class RouterInput(BaseModel):
    user_question: str= Field(description="This is the user's question, that needs to be categorized")


def router_node(state: State):
    """
    Router Node to choose the appropriate path for the user's question.
    
    This node evaluates the user's question using RAG scoring to determine relevance.
    If the question is relevant to RAG document (score >= threshold), it routes to RAG node.
    Otherwise, it routes to the generic LLM node.
    
    The node stores routing information in state fields ('category' and 'context') rather than
    in message additional_kwargs to keep messages clean.

    Validation: The input 'question' is validated using pydantic to ensure it is a string.
    
    Args:
        state (State): The current state containing:
            - messages: List with the user's question as the last message (role: 'user' or 'human')
    
    Returns:
        dict: State update containing:
            - current_question: List with the validated question string
            - category: "RAG relevant Issue" or "Not Related" (stored in state field)
            - context: Optional list of filtered context documents (only if category is "RAG relevant Issue")
            - messages: Optional error message dict (only if an error occurs during routing)
    """

    print("-> ROUTER ->")
    
    last_msg = state["messages"][-1]
    print("Last message in state:", last_msg)
    
    # Extract text content from the last message which is the question
    question = getattr(last_msg, 'content', last_msg)

    # Validating the input question
    valiadted_question = RouterInput(user_question= question)
    valiadted_question = valiadted_question.user_question

    print(f"\nRouter Node processing the question [{valiadted_question}] for routing")
    
    # Invoking the RAG score function to find if context retrieved is relevant to the user's question
    try:
        docs_and_scores = extract_docs_and_scores({"input": valiadted_question})
        print("\nRAG docs and scores: ", docs_and_scores)
        print("Docs and scores retrieved")
        #print("results_with_scores: ", docs_and_scores.get("results_with_scores", []))

        # Printing only scores
        print(docs_and_scores.get("scores", []))

    except Exception as e:
        # Capture full traceback for debugging and return safe fallback
        import traceback
        tb = traceback.format_exc()
        print("\n[router_node] Error while extracting docs and scores:\n", tb)
        print("\n[router_node] Routing to generic LLM due to error.")
        # Fallback: route to generic LLM
        return {
            "messages": [
                {
                    "role": "ai",
                    "content": "An error occurred while checking RAG relevance; routing to generic LLM.",
                    "category": "Not Related",
                }
            ]
        }

    # Decide routing using fused scores (if available) or presence of docs
    scores = docs_and_scores.get("scores", [])
    documents = docs_and_scores.get("documents", [])
    print(f"Documents retrieved: {documents}")
    print("Extracted scores and docs")


    threshold = float(os.getenv("RAG_SCORE_THRESHOLD", "0.60"))
    print(f"Using RAG score threshold: {threshold}")

    # Filter documents based on threshold - remove documents with scores below threshold
    filtered_documents = []
    filtered_scores = []
    for doc, score in zip(documents, scores):
        if score is not None and score >= threshold:
            filtered_documents.append(doc)
            filtered_scores.append(score)
    
    print(f"Filtered documents: {len(filtered_documents)} out of {len(documents)} (removed {len(documents) - len(filtered_documents)} below threshold)")

    context = None  # Initialize context variable
    top_rag_score = None  # Track for metrics
    if scores:
        top_score = max(scores)
        top_rag_score = top_score
        if top_score >= threshold:
            category = "RAG relevant Issue"

            # ── Stage 2: Cross-encoder re-ranking ──
            # Re-rank filtered documents using cross-encoder for higher precision
            rerank_top_n = int(os.getenv("RERANKER_TOP_N", "3"))
            try:
                reranked = rerank_documents(
                    query=valiadted_question,
                    documents=filtered_documents,
                    top_n=rerank_top_n,
                )
                context = [doc for doc, score in reranked]
                logger.info(
                    f"Re-ranked {len(filtered_documents)} → {len(context)} docs. "
                    f"Top cross-encoder score: {reranked[0][1]:.4f}" if reranked else ""
                )
            except Exception as e:
                logger.warning(f"Re-ranking failed, falling back to RRF-filtered docs: {e}")
                context = filtered_documents

            print(f"Top RAG score {top_score} exceeds threshold {threshold}, routing to RAG relevant Issue.")
        else:
            category = "Not Related"
            print(f"Top RAG score {top_score} below threshold {threshold}, routing to Not Related.")
    else:
        # No numeric scores available: fallback to presence of documents
        category = "Not Related"
        print("No RAG scores available, routing to Not Related by default.")

    # Store context and category in separate state fields to avoid additional_kwargs
    return_this = {
        "current_question": [
            valiadted_question  # Add the current question to state so that downstream nodes can access it
        ],
        "category": category,  # Store category in state field
        "top_rag_score": top_rag_score,  # For metrics tracking
    }

    # Only add context if it exists
    if context is not None:
        return_this["context"] = context

    return return_this


def route_question(state: State) -> str:
    """
    Routing function that determines which path the workflow should take.
    
    Reads the 'category' field from the state (set by router_node) and returns
    the appropriate route name for LangGraph conditional edges.
    
    Args:
        state (State): The current state containing:
            - category: String with routing decision ("RAG relevant Issue" or "Not Related")
    
    Returns:
        str: Either "RAG relevant Issue" or "Not Related" to route to the appropriate node
    """
    print("-> ROUTE QUESTION FUNCTION ->")
    
    # Read category from state field instead of additional_kwargs
    route_decision = state.get("category", "Not Related")
    
    # lower casing for safety
    route_decision = route_decision.lower() if isinstance(route_decision, str) else str(route_decision).lower()
    print(f"Routing decision from state: {route_decision}")
    if route_decision == "rag relevant issue": # has to be lower case!
        return "RAG relevant Issue"
    else:
        return "Not Related"