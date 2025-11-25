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

rag_prompt = read_prompt(filepath= os.environ["RAG_PROMPT_DIR"])

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
from utils.helpers import llm

# Importing both the BM25 and Vector Store retrievers - initialization is lazy loaded
from tools.document_loader import initialize_retrievers, vector_store_retriever, BM25_retriever

# Initialize retrievers on first use
initialize_retrievers()

def get_ensemble_retriever():
    """Get the ensemble retriever with only context
        Used for normal RAG retrieval, assuming the retrieved documents are relevant.
    """
    from tools.document_loader import vector_store_retriever, BM25_retriever
    from langchain.retrievers import EnsembleRetriever
    
    if vector_store_retriever is None or BM25_retriever is None:
        raise RuntimeError("Retrievers could not be initialized. Check document loader configuration.")
    
    return EnsembleRetriever(
        retrievers=[BM25_retriever, vector_store_retriever],
        weights=[0.3, 0.7]
    )


# Create ensemble retriever with only context
ensemble_retriever = get_ensemble_retriever()

# This is old hybrid search RAG pipeline without context
# LEGACY - kept for reference
hybrid_search_rag_pipeline = (
    {"context": RunnablePassthrough() | ensemble_retriever, "user_query": RunnablePassthrough()}
    |
    prompt
    |
    llm
    |
    parser
)

hybrid_search_rag_pipeline_with_context = (
    {"context": RunnablePassthrough() | ensemble_retriever, "user_query": RunnablePassthrough(), "messages": RunnablePassthrough()}
    | RunnablePassthrough.assign(
        response=(
            lambda x: (prompt | llm | parser).invoke(x)
        )
    )
)
