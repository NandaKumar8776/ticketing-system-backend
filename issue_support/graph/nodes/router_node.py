from memory.state import State
from pydantic import BaseModel, Field
from tools.rag_score import extract_docs_and_scores
from langchain_core.messages import AIMessage
import os

class RouterInput(BaseModel):
    user_question: str= Field(description="This is the user's question, that needs to be categorized")


def router_node(state: State):
    """
    Router Node to choose the appropriate path for the user's question.
    Does this by calling RAG score for the user question + previous messages history to check relevance.
    If relevant to PC Issue troubleshooting, routes to RAG node, else to generic LLM node.

    Validation: The input 'question' is validated using pydantic to ensure it is a string. 'messages' is assumed to be valid as it is managed by the State.
    Input: State with 'messages' containing the user's question as content and role as the last message. ex. [{'role': 'user', 'content': 'Hiii', ...}], user question is extracted from the last message.
    Output: State with 'messages' containing the role (assistant).
            Stores 'category' and 'context' (if available) in separate state fields to avoid additional_kwargs.
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
    if scores:
        top_score = max(scores)
        if top_score >= threshold:
            category = "PC Issue"
            context = filtered_documents  # Use filtered documents instead of all documents
            #reason = f"RAG can have relevant info to solve this issue routing to RAG call workflow."
            print(f"Top RAG score {top_score} exceeds threshold {threshold}, routing to PC Issue.")
        else:
            category = "Not Related"
            #reason = f"RAG does not have relevant info to solve this issue, so routing to normal LLM call workflow."
            print(f"Top RAG score {top_score} below threshold {threshold}, routing to Not Related.")
    else:
        # No numeric scores available: fallback to presence of documents
        category = "Not Related"
        #reason = f"RAG does not have relevant info to solve this issue, so routing to normal LLM call workflow."
        print("No RAG scores available, routing to Not Related by default.")

    # Store context and category in separate state fields to avoid additional_kwargs
    return_this = {
        "current_question": [
            valiadted_question  # Add the current question to state so that downstream nodes can access it
        ],
        "category": category  # Store category in state field
    }
    
    # Only add context if it exists
    if context is not None:
        return_this["context"] = context


    # Return state dict with routing decision as an AIMessage with/ without context based on the route
    return return_this


def route_question(state: State) -> str:
    """
    Routing function that returns which path the question should take.
    
    Input:
        state (State): The current state containing the category decision in the state field.
    Returns:
        str: Either "PC Issue" or "Not Related" to route to appropriate node
    """
    print("-> ROUTE QUESTION FUNCTION ->")
    
    # Read category from state field instead of additional_kwargs
    route_decision = state.get("category", "Not Related")
    
    # lower casing for safety
    route_decision = route_decision.lower() if isinstance(route_decision, str) else str(route_decision).lower()
    print(f"Routing decision from state: {route_decision}")
    if route_decision == "pc issue":
        return "PC Issue"
    else:
        return "Not Related"