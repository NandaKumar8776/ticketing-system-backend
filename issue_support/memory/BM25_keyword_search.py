# Setting up the BM25 Keyword Sparse Vector Search

from langchain_community.retrievers.bm25 import BM25Retriever

def BM25_retriever(docs):
    return BM25Retriever.from_documents(docs)
