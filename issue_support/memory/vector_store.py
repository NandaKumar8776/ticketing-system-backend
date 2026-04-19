"""
Milvus Vector Store initialization with HNSW index.

HNSW (Hierarchical Navigable Small World) provides:
- Sub-millisecond approximate nearest neighbor search at scale
- Tunable accuracy/speed tradeoff via M and efConstruction params
- Production-standard index used by major vector DB deployments

Metric: L2 (Euclidean distance) — compatible with all-MiniLM-L6-v2 embeddings.
"""

import os
import logging

from langchain_milvus import Milvus
from pymilvus import Collection, MilvusException, connections, db, utility
from utils.helpers import embeddings

logger = logging.getLogger(__name__)

# Configuration from environment (with sensible defaults)
URI = os.getenv("MILVUS_URI", "http://localhost:19530")
DB_NAME = os.getenv("MILVUS_DB_NAME", "milvus_assignment_test")
COLLECTION_NAME = "HNSW_Index_PC_Troubleshooting_PDF"

# HNSW index parameters
# M=16: max edges per node (higher = more accurate, more memory)
# efConstruction=200: build-time search width (higher = better index quality)
HNSW_INDEX_PARAMS = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {
        "M": 16,
        "efConstruction": 200,
    },
}

flat_milvus_vector_store = None

try:
    # Parse host/port from URI
    uri_clean = URI.replace("http://", "").replace("https://", "")
    host, port = uri_clean.split(":")
    conn = connections.connect(host=host, port=int(port))

    try:
        existing_databases = db.list_database()

        if DB_NAME in existing_databases:
            logger.info(f"Database '{DB_NAME}' already exists.")
            db.using_database(DB_NAME)

            # Drop all collections for a clean re-index on startup
            collections = utility.list_collections()
            for collection_name in collections:
                collection = Collection(name=collection_name)
                collection.drop()
                logger.info(f"Collection '{collection_name}' dropped.")

            db.drop_database(DB_NAME)
            logger.info(f"Database '{DB_NAME}' dropped for re-creation.")

        db.create_database(DB_NAME)
        logger.info(f"Database '{DB_NAME}' created successfully.")

    except MilvusException as e:
        logger.error(f"Milvus database operation failed: {e}")

    # Create vector store with HNSW index
    flat_milvus_vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI, "db_name": DB_NAME},
        index_params=HNSW_INDEX_PARAMS,
        collection_name=COLLECTION_NAME,
        collection_description=(
            "PC troubleshooting knowledge base indexed with HNSW "
            "for sub-millisecond approximate nearest neighbor search."
        ),
        consistency_level="Strong",
    )

    logger.info(
        f"Milvus vector store initialized: collection={COLLECTION_NAME}, "
        f"index=HNSW (M=16, efConstruction=200), metric=L2"
    )

except Exception as e:
    logger.warning(f"Could not initialize Milvus vector store: {e}")
    logger.warning(f"Make sure Milvus is running on {URI}")
    flat_milvus_vector_store = None