# Setting up the BM25 Keyword Sparse Vector Search

from langchain_community.retrievers.bm25 import BM25Retriever

def BM25_retriever(docs):
    """
    Create a BM25 keyword-based retriever from documents.
    
    BM25 is a sparse retrieval method that uses keyword matching with term frequency
    and inverse document frequency (TF-IDF) weighting. It's used in the ensemble
    retriever alongside vector search for hybrid retrieval.
    
    Args:
        docs (list): List of document objects to index.
    
    Returns:
        BM25Retriever: A retriever instance that can search documents by keywords.
    """
    return BM25Retriever.from_documents(docs)
