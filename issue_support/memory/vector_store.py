"""
Milvus Vector Store initialization with HNSW index.

Supports three deployment modes selected via MILVUS_URI:
  Local   http://host:port     Self-hosted Milvus (Docker Compose default)
  Cloud   https://...          Zilliz Cloud — set ZILLIZ_API_KEY as well
  Lite    ./path/to/file.db    Milvus Lite — embedded, no server required

HNSW (Hierarchical Navigable Small World) provides sub-millisecond ANN search
with a tunable accuracy/speed tradeoff. Metric: L2, compatible with all-MiniLM-L6-v2.
"""

import os
import logging

from langchain_milvus import Milvus
from utils.helpers import embeddings

logger = logging.getLogger(__name__)

# APP_MILVUS_URI takes priority — use this for Lite/Cloud URIs so pymilvus
# doesn't read a non-HTTP value from MILVUS_URI at import time.
URI = os.getenv("APP_MILVUS_URI") or os.getenv("MILVUS_URI", "http://localhost:19530")
DB_NAME = os.getenv("MILVUS_DB_NAME", "milvus_assignment_test")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY", "")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "IT_Support_Knowledge_Base")

HNSW_INDEX_PARAMS = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {"M": 16, "efConstruction": 200},
}

# Detect deployment mode from URI scheme
_is_lite = not URI.startswith("http")          # ./milvus.db or similar
_is_cloud = URI.startswith("https://")         # Zilliz Cloud
_is_local = URI.startswith("http://")          # Self-hosted Milvus

flat_milvus_vector_store = None

try:
    if _is_lite:
        # Milvus Lite — embedded file-based store, no server management needed
        logger.info(f"Milvus mode: Lite ({URI})")
        connection_args = {"uri": URI}

    elif _is_cloud:
        # Zilliz Cloud — HTTPS endpoint with token authentication
        logger.info(f"Milvus mode: Zilliz Cloud ({URI})")
        if not ZILLIZ_API_KEY:
            raise ValueError("ZILLIZ_API_KEY must be set when using a Zilliz Cloud URI")
        connection_args = {"uri": URI, "token": ZILLIZ_API_KEY}

    else:
        # Self-hosted Milvus — manage database lifecycle manually
        from pymilvus import Collection, MilvusException, connections, db, utility

        logger.info(f"Milvus mode: Local ({URI})")
        uri_clean = URI.replace("http://", "")
        host, port = uri_clean.split(":")
        connections.connect(host=host, port=int(port))

        try:
            existing_databases = db.list_database()
            if DB_NAME in existing_databases:
                db.using_database(DB_NAME)
                for col in utility.list_collections():
                    Collection(name=col).drop()
                    logger.info(f"Dropped collection '{col}'")
                db.drop_database(DB_NAME)
            db.create_database(DB_NAME)
            logger.info(f"Database '{DB_NAME}' created")
        except MilvusException as e:
            logger.error(f"Milvus database operation failed: {e}")

        connection_args = {"uri": URI, "db_name": DB_NAME}

    flat_milvus_vector_store = Milvus(
        embedding_function=embeddings,
        connection_args=connection_args,
        index_params=HNSW_INDEX_PARAMS,
        collection_name=COLLECTION_NAME,
        collection_description=(
            "PC troubleshooting knowledge base indexed with HNSW "
            "for sub-millisecond approximate nearest neighbor search."
        ),
        consistency_level="Strong",
        drop_old=True,
    )
    logger.info(f"Milvus vector store ready: collection={COLLECTION_NAME}")

except Exception as e:
    logger.warning(f"Could not initialize Milvus vector store: {e}")
    flat_milvus_vector_store = None