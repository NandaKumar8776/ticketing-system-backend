from utils.helpers import doc_output_formatter
from tools.ensemble_retriever_with_scores import EnsembleRetrieverWithScores

# Importing both the BM25 and Vector Store retrievers - initialization is lazy loaded
from tools.document_loader import initialize_retrievers, vector_store_retriever, BM25_retriever

# Initialize retrievers on first use
initialize_retrievers()

# Initialize the ensemble/ hybrid retriever with scores- for BM25 and Flat Indexed L2 Dense Vector Retriever
def get_ensemble_retriever_with_scores():
    """
    Get the ensemble retriever that returns documents with relevance scores.
    
    This retriever combines BM25 (keyword search) and Vector Store (semantic search)
    retrievers with weighted scores. The scores can be used to determine if RAG
    is appropriate for a given query (e.g., by comparing against a threshold).
    
    Returns:
        EnsembleRetrieverWithScores: An ensemble retriever that returns (document, score) tuples.
            - BM25 weight: 0.3
            - Vector Store weight: 0.7
    
    Raises:
        RuntimeError: If retrievers are not initialized.
    """
    from tools.document_loader import vector_store_retriever, BM25_retriever
    
    if vector_store_retriever is None or BM25_retriever is None:
        raise RuntimeError("Retrievers could not be initialized. Check document loader configuration.")
    
    return EnsembleRetrieverWithScores(
        retrievers=[BM25_retriever, vector_store_retriever],
        weights=[0.3, 0.7]
    )

# Create ensemble retriever with scores
ensemble_retriever_with_scores = get_ensemble_retriever_with_scores()



def extract_docs_and_scores(input_dict):
    """
    Extract documents and scores from ensemble retriever results.
    
    This function processes the output from get_ensemble_retriever_with_scores()
    and separates documents from their relevance scores. Used by router_node to
    determine if a query should be routed to RAG based on score thresholds.
    
    Args:
        input_dict (dict or str): Input containing the search query.
            If dict, expects 'input' key with the query string.
            If str, uses the string directly as the query.
        
    Returns:
        dict: Dictionary containing:
            - documents (list): List of retrieved document objects
            - scores (list): List of relevance scores (may contain None values)
            - context_str (str): Formatted string of all document contents
            - results_with_scores (list): Raw results from the retriever
    """
    query = input_dict if isinstance(input_dict, str) else input_dict.get("input", "")
    
    # Get results with scores
    results_with_scores = ensemble_retriever_with_scores.invoke(query)
    
    # Optional debug logging to inspect result shapes
    import os
    if os.getenv("RAG_DEBUG", "false").lower() == "true":
        print("[rag_score] raw results_with_scores:")
        for i, item in enumerate(results_with_scores):
            try:
                print(i, type(item), repr(item))
            except Exception:
                print(i, type(item), str(item))
    
    # Separate documents and scores
    documents = []
    scores = []
    for item in results_with_scores:
        # item may be (doc, score) or (doc, score, ...extra)
        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                doc = item[0]
                score = item[1]
            elif len(item) == 1:
                doc = item[0]
                score = None
            else:
                continue
        else:
            # unexpected shape: treat item as document
            doc = item
            score = None

        documents.append(doc)
        scores.append(score)
    
    # Format context as before (for compatibility)
    context_str = doc_output_formatter(documents) if documents else ""
    
    return {
        "documents": documents,
        "scores": scores,
        "context_str": context_str,
        "results_with_scores": results_with_scores  # Include full results if needed
    }

