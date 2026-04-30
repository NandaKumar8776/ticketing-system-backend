import os

## Parser with Pydantic input checking and output checking

from pydantic import BaseModel,Field

class OutputCheck(BaseModel):
    output: str= Field(
        description="The output of the RAG Pipeline, with the context in mind"
    )

from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object= OutputCheck)

## Prompt Setup

from langchain_core.prompts import ChatPromptTemplate
from utils.helpers import read_prompt

rag_prompt = read_prompt(filepath=os.environ.get("RAG_PROMPT_DIR", "prompts/rag_prompt.txt"))

prompt = ChatPromptTemplate(
    messages= [
        ("system", rag_prompt),
        ("human","{user_query}"),
    ],
    input_variables=["user_query","messages", "context"],
    partial_variables= {"output_structure": parser.get_format_instructions()}
)

### Hybrid Search RAG Pipeline

from langchain_core.runnables import RunnablePassthrough
from utils.helpers import rag_llm

# Importing both the BM25 and Vector Store retrievers - initialization is lazy loaded
from tools.document_loader import initialize_retrievers, vector_store_retriever, BM25_retriever

# Initialize retrievers on first use
initialize_retrievers()

def get_ensemble_retriever():
    """
    Get the ensemble retriever for standard RAG retrieval.
    
    This retriever combines BM25 (keyword search) and Vector Store (semantic search)
    retrievers. It returns documents without scores, assuming retrieved documents
    are relevant. Used by the RAG pipeline when context is already validated.
    
    Note: This is different from get_ensemble_retriever_with_scores() which is
    used for routing decisions. This retriever is used when we already know RAG
    is appropriate for the query.
    
    Returns:
        EnsembleRetriever: An ensemble retriever that returns documents.
            - BM25 weight: 0.3
            - Vector Store weight: 0.7
    
    Raises:
        RuntimeError: If retrievers are not initialized.
    """
    from tools.document_loader import vector_store_retriever, BM25_retriever
    from langchain.retrievers.ensemble import EnsembleRetriever

    if BM25_retriever is None:
        raise RuntimeError("Knowledge base is empty. POST /ingest to load documents first.")

    if vector_store_retriever is None:
        return EnsembleRetriever(retrievers=[BM25_retriever], weights=[1.0])

    retriever = EnsembleRetriever(
        retrievers=[BM25_retriever, vector_store_retriever],
        weights=[0.3, 0.7],
    )
    print("\n[rag_hybrid_retriever] Ensemble retriever created with BM25 and Vector Store retrievers.")
    return retriever


# Context from the router node is passed directly to the RAG pipeline now
hybrid_search_rag_pipeline_with_context = (
    {
        "context": RunnablePassthrough(),
        "user_query": RunnablePassthrough(),
        "messages": RunnablePassthrough()
    }
    | RunnablePassthrough.assign(
        response=lambda x: (prompt | rag_llm | parser).invoke(x)
    )
)

