from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import TesseractBlobParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

file_path = os.getenv("FILE_DIR", "")
print(f"\n[Document Loader] Starting- checking if already initialized...")
image_parser = TesseractBlobParser()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# Initialize retrievers as None - will be populated on first use
vector_store_retriever = None
BM25_retriever = None
_initialized = False

# Tracks all docs ever loaded — used to rebuild BM25 on new ingestion
_all_docs = []


def _load_and_chunk(pdf_path: str) -> list:
    """Load a PDF file and return chunked documents."""
    loader = PyMuPDFLoader(
        file_path=pdf_path,
        mode="page",
        images_parser=image_parser,
        images_inner_format="html-img",
        extract_tables="markdown",
    )
    docs = loader.load()

    filename = os.path.basename(pdf_path)
    for doc in docs:
        doc.page_content = doc.page_content.strip()
        # Explicitly build a fixed metadata schema so Milvus always sees the same fields.
        # PyMuPDF adds variable PDF fields (producer, creator, etc.) that differ per page
        # and would cause DataNotMatchException when Milvus enforces the inferred schema.
        doc.metadata = {
            "page": int(doc.metadata.get("page", 0)),
            "ingested_file": filename,
        }

    return text_splitter.split_documents(docs)


def _upload_to_vector_store(docs: list) -> bool:
    """Upload docs to Milvus. Returns True on success."""
    from memory.vector_store import flat_milvus_vector_store

    if flat_milvus_vector_store is None:
        print("\n[Document Loader] WARNING: Milvus vector store is not available!")
        return False

    try:
        # Reconnect before inserting — Milvus Lite's gRPC connection can go stale
        # after ~10-15s of inactivity (e.g. during PDF loading), causing
        # ConnectionNotExistException. Reconnecting is safe and idempotent.
        from pymilvus import connections as _milvus_connections
        _uri = os.getenv("APP_MILVUS_URI") or os.getenv("MILVUS_URI", "http://localhost:19530")
        _alias = getattr(flat_milvus_vector_store, "alias", "default")
        if _uri.startswith("http://"):
            _db_name = os.getenv("MILVUS_DB_NAME", "milvus_assignment_test")
            _milvus_connections.connect(alias=_alias, uri=_uri, db_name=_db_name)
        else:
            _milvus_connections.connect(alias=_alias, uri=_uri)

        flat_milvus_vector_store.add_documents(documents=docs)
        return True
    except Exception as e:
        print(f"\n[Document Loader] ERROR uploading to vector store: {e}")
        return False


def _rebuild_bm25(docs: list):
    """Rebuild the BM25 retriever from the full document corpus."""
    global BM25_retriever
    from memory.BM25_keyword_search import BM25_retriever as create_bm25_retriever
    BM25_retriever = create_bm25_retriever(docs=docs)

    # Propagate updated retriever into ensemble retrievers that hold a reference
    try:
        import tools.rag_score as rag_score
        if hasattr(rag_score, "ensemble_retriever_with_scores"):
            rag_score.ensemble_retriever_with_scores.retrievers[0] = BM25_retriever
    except Exception:
        pass


def _index_pdfs(pdf_paths: list):
    """Chunk and index a list of PDF paths into Milvus + BM25."""
    global vector_store_retriever, _all_docs

    for pdf_path in pdf_paths:
        try:
            docs = _load_and_chunk(pdf_path)
            _all_docs.extend(docs)
            print(f"\n[Document Loader] Loaded: {os.path.basename(pdf_path)} ({len(docs)} chunks)")
        except Exception as e:
            print(f"\n[Document Loader] ERROR loading {os.path.basename(pdf_path)}: {e}")

    if not _all_docs:
        return

    print(f"\n[Document Loader] Indexing {len(_all_docs)} total chunks into vector store...")
    if _upload_to_vector_store(_all_docs):
        from memory.vector_store import flat_milvus_vector_store
        vector_store_retriever = flat_milvus_vector_store.as_retriever()
        print("\n[Document Loader] Vector Store is ready")
    else:
        vector_store_retriever = None

    _rebuild_bm25(_all_docs)
    print("\n[Document Loader] BM25 is ready")


def initialize_retrievers():
    """
    Called once at startup via FastAPI lifespan.

    Priority order:
      1. GCS (if GCS_BUCKET is set): downloads all PDFs from gs://<bucket>/documents/
      2. FILE_DIR fallback: loads the single configured PDF (local dev)
      3. Empty start: if neither yields documents, API starts with no knowledge base
    """
    global _initialized

    if _initialized:
        return

    print("\n[Document Loader] Initializing retrievers...")

    pdf_paths = []

    # --- GCS: primary source for production ---
    from utils.gcs_store import is_configured, download_all_documents
    if is_configured():
        upload_dir = os.getenv("UPLOAD_DIR", "/app/data/uploads")
        print(f"\n[Document Loader] Pulling documents from GCS...")
        pdf_paths = download_all_documents(upload_dir)
        if pdf_paths:
            print(f"\n[Document Loader] Found {len(pdf_paths)} document(s) in GCS")
        else:
            print("\n[Document Loader] GCS bucket is empty — no documents to load")

    # --- FILE_DIR: fallback for local dev ---
    if not pdf_paths and file_path and os.path.exists(file_path):
        print(f"\n[Document Loader] Loading from FILE_DIR: {file_path}")
        pdf_paths = [file_path]

    if not pdf_paths:
        print("\n[Document Loader] Starting with empty knowledge base — use POST /ingest to load documents.")
        _initialized = True
        return

    _index_pdfs(pdf_paths)
    _initialized = True


def ingest_pdf(pdf_path: str) -> dict:
    """
    Ingest a new PDF into the retrieval system at runtime.

    Chunks the PDF, appends to Milvus, rebuilds BM25, then uploads the
    original file to GCS so it persists across container restarts.

    Args:
        pdf_path: Absolute path to the PDF file on the container filesystem.

    Returns:
        dict with keys: filename, num_chunks, total_corpus_chunks, success,
                        gcs_persisted, error (if any)
    """
    global vector_store_retriever, _all_docs

    filename = os.path.basename(pdf_path)
    print(f"\n[Document Loader] Ingesting new file: {filename}")

    try:
        docs = _load_and_chunk(pdf_path)
    except Exception as e:
        return {"filename": filename, "num_chunks": 0, "success": False,
                "gcs_persisted": False, "error": str(e)}

    num_chunks = len(docs)
    print(f"\n[Document Loader] {filename} → {num_chunks} chunks")

    if not _upload_to_vector_store(docs):
        return {"filename": filename, "num_chunks": num_chunks, "success": False,
                "gcs_persisted": False, "error": "Failed to upload to Milvus vector store"}

    from memory.vector_store import flat_milvus_vector_store
    vector_store_retriever = flat_milvus_vector_store.as_retriever()

    _all_docs.extend(docs)
    _rebuild_bm25(_all_docs)

    # Persist to GCS so the document survives container restarts.
    # If GCS is configured and the upload fails, treat the whole ingest as failed:
    # the document is in Milvus for this session but will be lost on restart.
    from utils.gcs_store import is_configured, upload_document
    gcs_persisted = False
    if is_configured():
        gcs_persisted = upload_document(pdf_path)
        if not gcs_persisted:
            # Roll back: remove the chunks we just added so state stays consistent
            _all_docs[-num_chunks:] = []
            return {
                "filename": filename,
                "num_chunks": num_chunks,
                "total_corpus_chunks": len(_all_docs),
                "success": False,
                "gcs_persisted": False,
                "error": "GCS upload failed — document not persisted. Check GCS permissions and retry.",
            }
        print(f"\n[Document Loader] {filename} persisted to GCS")

    print(f"\n[Document Loader] Ingestion complete: {filename} ({num_chunks} chunks, "
          f"corpus now {len(_all_docs)} total chunks)")

    return {
        "filename": filename,
        "num_chunks": num_chunks,
        "total_corpus_chunks": len(_all_docs),
        "success": True,
        "gcs_persisted": gcs_persisted,
        "error": None,
    }
