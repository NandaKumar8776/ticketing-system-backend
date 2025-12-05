from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import TesseractBlobParser
import os

file_path = os.environ["FILE_DIR"]
print(f"\n[Document Loader] Starting- checking if already initialized...")
image_parser = TesseractBlobParser()

# Initialize retrievers as None - will be populated on first use
vector_store_retriever = None
BM25_retriever = None
_initialized = False


def initialize_retrievers():
    """
    Lazy initialization of retrievers. This is called on first use to avoid
    connecting to Milvus during module import.
    """
    global vector_store_retriever, BM25_retriever, _initialized
    
    if _initialized:
        return
    
    print("\n[Document Loader] Initializing retrievers...")
    
    #### Loading PDF with multi-modal data

    loader = PyMuPDFLoader(
        file_path=file_path,
        mode="page",
        images_parser= image_parser,
        images_inner_format= "html-img",
        extract_tables="markdown", # optional. takes a lot of time

    )

    docs = loader.load()

    # Cleaning headers with RE, since the source document needs it
    import re

    def clean_page_text(text: str) -> str:
        # Remove headers like "Page X of Y" if needed
        return text.strip()

    for doc in docs:
        doc.page_content = clean_page_text(doc.page_content)

    print("\n[Document Loader] Loaded the data")

    #### Utilizing Recursive Character Text Splitter so we achieve overlapping of texts between chunks


    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,#hyperparameter
        chunk_overlap=50 #hyperparemeter
    )

    print("\n[Document Loader] Chunking started")

    docs = text_splitter.split_documents(docs)

    print("\n[Document Loader] Chunking is done")

    #### Load documents into the vector store


    from memory.vector_store import flat_milvus_vector_store

    if flat_milvus_vector_store is None:
        print("\n[Document Loader] WARNING: Milvus vector store is not available!")
        vector_store_retriever = None
    else:
        print("\n[Document Loader] Uploading the chunked docs to the vector store")
        try:
            flat_milvus_vector_store.add_documents(
                documents=docs
            )

            vector_store_retriever = flat_milvus_vector_store.as_retriever()
            print("\n[Document Loader] Vector Store is ready")
        except Exception as e:
            print(f"\n[Document Loader] ERROR uploading to vector store: {e}")
            vector_store_retriever = None


    #### Load documents into the Keyword Search BM25


    from memory.BM25_keyword_search import BM25_retriever as create_bm25_retriever

    print("\n[Document Loader] Uploading the chunked docs to the BM25 Keyword Search")

    BM25_retriever = create_bm25_retriever(docs=docs)

    print("\n[Document Loader] BM25 Keyword Search is ready")
    
    _initialized = True

