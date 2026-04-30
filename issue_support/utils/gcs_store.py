"""
GCS document store — persistent backing for the knowledge base.

All uploaded PDFs are stored under gs://<GCS_BUCKET>/documents/.
On container startup, initialize_retrievers() downloads every PDF from
this prefix so the knowledge base survives container restarts.

Not used when GCS_BUCKET is unset (local dev falls back to FILE_DIR).
"""

import os
import logging

logger = logging.getLogger(__name__)

_BUCKET = os.getenv("GCS_BUCKET", "")
_PREFIX = "documents/"

# Module-level singleton — avoids re-establishing the GCS connection on every call.
_gcs_client = None


def _get_client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client


def is_configured() -> bool:
    return bool(_BUCKET)


def upload_document(local_path: str) -> bool:
    """Upload a PDF to gs://<GCS_BUCKET>/documents/<filename>."""
    if not is_configured():
        return False
    try:
        client = _get_client()
        blob = client.bucket(_BUCKET).blob(f"{_PREFIX}{os.path.basename(local_path)}")
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {os.path.basename(local_path)} to gs://{_BUCKET}/{_PREFIX}")
        return True
    except Exception as e:
        logger.error(f"GCS upload failed: {e}")
        return False


def download_all_documents(local_dir: str) -> list:
    """
    Download all PDFs from gs://<GCS_BUCKET>/documents/ to local_dir.
    Returns list of local file paths that were downloaded.
    """
    if not is_configured():
        return []
    try:
        client = _get_client()
        blobs = list(client.bucket(_BUCKET).list_blobs(prefix=_PREFIX))

        os.makedirs(local_dir, exist_ok=True)
        downloaded = []
        for blob in blobs:
            if not blob.name.lower().endswith(".pdf"):
                continue
            filename = os.path.basename(blob.name)
            local_path = os.path.join(local_dir, filename)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded {filename} from GCS")
            downloaded.append(local_path)
        return downloaded
    except Exception as e:
        logger.error(f"GCS download failed: {e}")
        return []
