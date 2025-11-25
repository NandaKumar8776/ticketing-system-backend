from utils.helpers import doc_output_formatter
from tools.ensemble_retriever_with_scores import EnsembleRetrieverWithScores

# Importing both the BM25 and Vector Store retrievers - initialization is lazy loaded
from tools.document_loader import initialize_retrievers, vector_store_retriever, BM25_retriever

# Initialize retrievers on first use
initialize_retrievers()

# Initialize the ensemble/ hybrid retriever with scores- for BM25 and Flat Indexed L2 Dense Vector Retriever
def get_ensemble_retriever_with_scores():
    """Get the ensemble retriever with scores
    Can be used to determine if RAG is good for the query or not based on scores.
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
    Helper function to extract documents and scores from ensemble retriever results.
    Returns both the documents (for compatibility) and the scores separately.
    
    Args:
        input_dict: Dict containing 'input' key with search query
        
    Returns:
        Dict with 'documents', 'scores', and 'context_str' keys
    """
    query = input_dict if isinstance(input_dict, str) else input_dict.get("input", "")
    
    # Get results with scores
    results_with_scores = ensemble_retriever_with_scores.invoke(query)
    
    # Separate documents and scores
    documents = [doc for doc, score in results_with_scores]
    scores = [score for doc, score in results_with_scores]
    
    # Format context as before (for compatibility)
    context_str = doc_output_formatter(documents) if documents else ""
    
    return {
        "documents": documents,
        "scores": scores,
        "context_str": context_str,
        "results_with_scores": results_with_scores  # Include full results if needed
    }

