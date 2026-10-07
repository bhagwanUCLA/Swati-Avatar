"""
server.py
---------
FastAPI backend for the Portfolio RAG pipeline.

Auth
----
Admin routes require a JWT Bearer token obtained from POST /login.
Password is verified once (Argon2 against Firestore hash); a signed JWT
(30-min expiry) is returned. Subsequent requests verify the cheap JWT
signature instead. JWT secret = sha256(admin_hash) — stable across gunicorn
workers and automatically invalidated if the password changes.

Streaming architecture
----------------------
The Anthropic SDK is synchronous and blocking.  Running it directly inside
an `async def` would freeze the entire uvicorn event loop for the duration
of each Claude call, causing gunicorn WORKER TIMEOUT on longer queries.

Solution: _run_llm_in_thread() submits the sync stream_answer generator to a
ThreadPoolExecutor.  The generator puts ('token', text), ('done', answer),
or ('error', msg) items into a thread-safe queue.Queue.  The async
event_stream() coroutine polls that queue with short sleeps, keeping the
event loop free for other requests and heartbeat keepalives.

Gunicorn start command (Cloud Run):
  gunicorn -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120

Session persistence
-------------------
If GOOGLE_CLOUD_PROJECT is set, chat histories are stored in Firestore
(collection: rag_sessions).  Otherwise an in-memory dict is used — fine
for local development, but histories are lost on restart.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import functools
import hashlib
import inspect
import hmac
import json
import logging
import os
import secrets
import queue as _sync_queue
import socket
import threading
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import requests as _req

import jwt
from typing import Annotated, AsyncGenerator, Optional
from pathlib import Path

from fastapi import Depends, FastAPI, UploadFile, File, Form, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, PlainTextResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pwdlib import PasswordHash
from pydantic import BaseModel, Field
import tempfile
import zipfile


from orchestrator import RAGOrchestrator
from rag_query import RAG
from dotenv import load_dotenv

# Google Drive imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
import io

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolio RAG API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for blocking Anthropic SDK calls
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)


# ---------------------------------------------------------------------------
# GCS FAISS index persistence
# ---------------------------------------------------------------------------

_GCS_INDEX_FILES = ["faiss.index", "metadata.pkl"]
_GCS_INDEX_PREFIX = "rag_index"
_GCS_INDEX_MANIFEST = f"{_GCS_INDEX_PREFIX}/current.json"


def _gcs_client():
    """Lazy import — google-cloud-storage only needed in prod."""
    from google.cloud import storage
    return storage.Client()


def _download_index_from_gcs(bucket_name: str, index_dir: str) -> bool:
    """
    Download faiss.index + metadata.pkl from GCS into index_dir.
    Returns True if both files were found and downloaded.
    """
    try:
        client  = _gcs_client()
        bucket  = client.bucket(bucket_name)
        path    = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        manifest_blob = bucket.blob(_GCS_INDEX_MANIFEST)
        if manifest_blob.exists():
            manifest = json.loads(manifest_blob.download_as_text())
            release_id = manifest.get("release_id")
            files = manifest.get("files")
            if not release_id or files != _GCS_INDEX_FILES:
                raise ValueError("GCS index manifest is missing a valid release ID or file list.")
            blob_paths = [f"{_GCS_INDEX_PREFIX}/releases/{release_id}/{fname}" for fname in _GCS_INDEX_FILES]
        else:
            logger.info("GCS: no index manifest found; loading legacy index files.")
            blob_paths = [f"{_GCS_INDEX_PREFIX}/{fname}" for fname in _GCS_INDEX_FILES]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            for fname, blob_path in zip(_GCS_INDEX_FILES, blob_paths):
                blob = bucket.blob(blob_path)
                if not blob.exists():
                    logger.warning("GCS: %s not found in bucket %s", blob_path, bucket_name)
                    return False
                blob.download_to_filename(str(temp_path / fname))

            for fname in _GCS_INDEX_FILES:
                os.replace(temp_path / fname, path / fname)
                logger.info("GCS ↓ downloaded %s", fname)
        return True
    except Exception as exc:
        logger.error("GCS download failed: %s", exc)
        return False


def _upload_index_to_gcs(bucket_name: str, index_dir: str) -> None:
    """
    Upload a consistent FAISS index release and publish it with a manifest.
    Raises when either index file or the manifest cannot be persisted.
    """
    path = Path(index_dir)
    missing_files = [fname for fname in _GCS_INDEX_FILES if not (path / fname).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Cannot upload incomplete FAISS index: {', '.join(missing_files)}")

    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    release_id = uuid.uuid4().hex
    release_prefix = f"{_GCS_INDEX_PREFIX}/releases/{release_id}"

    try:
        for fname in _GCS_INDEX_FILES:
            bucket.blob(f"{release_prefix}/{fname}").upload_from_filename(str(path / fname))
            logger.info("GCS ↑ uploaded %s for index release %s", fname, release_id)

        manifest = {
            "release_id": release_id,
            "files": _GCS_INDEX_FILES,
            "published_at": _sync_timestamp(),
        }
        bucket.blob(_GCS_INDEX_MANIFEST).upload_from_string(
            json.dumps(manifest), content_type="application/json"
        )
        logger.info("GCS ↑ published index release %s", release_id)
    except Exception as exc:
        logger.error("GCS upload failed for index release %s: %s", release_id, exc)
        raise RuntimeError(f"GCS index upload failed: {exc}") from exc


def _download_system_prompt_from_gcs(bucket_name: str, config_dir: str) -> bool:
    """
    Download system_prompt.txt from GCS into config_dir.
    Returns True if downloaded successfully.
    """
    try:
        client = _gcs_client()
        bucket = client.bucket(bucket_name)
        path = Path(config_dir)
        path.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob("system_config/system_prompt.txt")
        if blob.exists():
            blob.download_to_filename(str(path / "system_prompt.txt"))
            logger.info("GCS ↓ downloaded system_prompt.txt")
            return True
        else:
            logger.warning("GCS: system_prompt.txt not found in bucket %s", bucket_name)
            return False
    except Exception as exc:
        logger.error("GCS system prompt download failed: %s", exc)
        return False


def _upload_system_prompt_to_gcs(bucket_name: str, config_dir: str) -> bool:
    """
    Upload system_prompt.txt from config_dir to GCS.
    Returns True if uploaded successfully.
    """
    try:
        client = _gcs_client()
        bucket = client.bucket(bucket_name)
        path = Path(config_dir) / "system_prompt.txt"
        if path.exists():
            bucket.blob("system_config/system_prompt.txt").upload_from_filename(str(path))
            logger.info("GCS ↑ uploaded system_prompt.txt")
            return True
        else:
            logger.warning("GCS upload: system_prompt.txt not found locally")
            return False
    except Exception as exc:
        logger.error("GCS system prompt upload failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Admin auth dependency
# ---------------------------------------------------------------------------

_http_bearer = HTTPBearer(auto_error=False)


def require_admin(
    creds: Annotated[Optional[HTTPAuthorizationCredentials], Depends(_http_bearer)],
) -> None:
    """
    FastAPI dependency that enforces Bearer token auth on admin routes.
    Verifies the JWT obtained from POST /login.
    """
    global _cached_admin_hash

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        return

    if not _cached_admin_hash:
        try:
            _cached_admin_hash = _get_admin_hash_from_db()
        except Exception as exc:
            logger.error("require_admin: Firestore read failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cannot reach database. Check Firestore IAM permissions.",
            )

    if not _cached_admin_hash:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin password is not configured. Please complete setup.",
        )

    # Always recalculate secret from current hash (don't cache) to avoid multi-worker issues
    jwt_secret = hashlib.sha256(_cached_admin_hash.encode()).hexdigest()

    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        jwt.decode(creds.credentials, jwt_secret, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Shorthand type alias used in admin route signatures
AdminDep = Annotated[None, Depends(require_admin)]


# ---------------------------------------------------------------------------
# Session store — Firestore (prod) or in-memory dict (local dev)
# ---------------------------------------------------------------------------

def _build_session_store():
    """
    Returns a FirestoreSessionStore if GOOGLE_CLOUD_PROJECT is set,
    otherwise returns an InMemorySessionStore.  Both expose the same
    interface so rag_query.RAG doesn't care which one it gets.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if project:
        try:
            from firestore_sessions import FirestoreSessionStore
            store = FirestoreSessionStore(project=project)
            logger.info("Session store: Firestore (project=%s)", project)
            return store
        except Exception as exc:
            logger.warning(
                "Firestore unavailable (%s) — falling back to in-memory sessions.", exc
            )
    from firestore_sessions import InMemorySessionStore
    logger.info("Session store: in-memory (local dev mode)")
    return InMemorySessionStore()


_session_store = _build_session_store()


# ---------------------------------------------------------------------------
# Admin Auth Hashing & Storage
# ---------------------------------------------------------------------------

password_hasher = PasswordHash.recommended()
_cached_admin_hash: Optional[str] = None
_jwt_secret: Optional[str] = None

def _firestore_client():
    """Return a Firestore client using GOOGLE_CLOUD_PROJECT and FIRESTORE_DB env vars."""
    from google.cloud import firestore
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    database = os.environ.get("FIRESTORE_DB", "(default)")
    return firestore.Client(project=project, database=database)


def _get_admin_hash_from_db() -> Optional[str]:
    """
    Returns the stored hash, or None if the document doesn't exist.
    Raises on any Firestore connection / permission error so callers
    can distinguish "not set" from "DB unreachable".
    """
    if not os.environ.get("GOOGLE_CLOUD_PROJECT", ""):
        return None
    db = _firestore_client()
    doc = db.collection("system_config").document("admin").get()
    if doc.exists:
        return doc.to_dict().get("password_hash")
    return None  # document absent = password genuinely not configured yet


def _set_admin_hash_in_db(hashed_pwd: str) -> None:
    """
    Persists the Argon2 hash to Firestore.
    Raises on failure — callers must handle and return an error response.
    """
    if not os.environ.get("GOOGLE_CLOUD_PROJECT", ""):
        return
    db = _firestore_client()
    db.collection("system_config").document("admin").set(
        {"password_hash": hashed_pwd}, merge=True
    )


async def _send_reset_email(reset_url: str) -> None:
    """
    Send password reset email via Mailjet.
    Raises RuntimeError if env vars not configured.
    """
    import httpx
    api_key    = os.environ.get("MAILJET_API_KEY", "")
    secret_key = os.environ.get("MAILJET_SECRET_KEY", "")
    from_email = os.environ.get("MAILJET_FROM_EMAIL", "")
    to_email   = os.environ.get("ADMIN_EMAIL", "")
    if not all([api_key, secret_key, from_email, to_email]):
        raise RuntimeError("Mailjet env vars not configured.")
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://api.mailjet.com/v3.1/send",
            auth=(api_key, secret_key),
            json={
                "Messages": [{
                    "From": {"Email": from_email, "Name": "2Meditate Admin"},
                    "To":   [{"Email": to_email}],
                    "Subject": "Admin Panel — Password Reset",
                    "HTMLPart": f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family: Arial, sans-serif; line-height: 1.6;">
  <p>You requested a password reset for the 2Meditate Admin panel.</p>
  <p><a href="{reset_url}" style="color: #2c5530; font-weight: bold; text-decoration: none; background: #f0f0f0; padding: 10px 20px; display: inline-block; border-radius: 5px;">Reset Your Password</a></p>
  <p>Or copy and paste this link in your browser:</p>
  <p><code style="background: #f0f0f0; padding: 10px; display: block; word-break: break-all;">{reset_url}</code></p>
  <p>This link expires in 10 minutes and can only be used once.</p>
  <p>If you did not request this, ignore this email.</p>
</body>
</html>""",
                }]
            },
        )
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "gemini_api_key":    os.environ.get("GEMINI_API_KEY", ""),
    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
    "youtube_api_key":   os.environ.get("YOUTUBE_API_KEY", ""),
    "hf_model_name":     "gemini-embedding-001",
    "chunk_size":        5000,
    "chunk_overlap":     300,
    "dedup_threshold":   None,
    "min_tokens":        20,
    "index_dir":         "./rag_index",
    "cache_dir":         "./scraper_cache",
    "follow_external":   True,
    "device":            "cpu",
    "model":             "claude-sonnet-4-6",
    "top_k":             6,
    "gcs_bucket":        os.environ.get("GCS_BUCKET", ""),  # e.g. "swati-rag-index"
}

_current_config: dict             = dict(_DEFAULT_CONFIG)
_rag:            Optional[RAGOrchestrator] = None

# Model cache: {models: [...], timestamp: ...}
_model_cache: dict = {"models": None, "timestamp": None}
_MODEL_CACHE_TTL = 300  # 5 minutes

# Password reset rate limiting: {ip: [timestamp, ...]}
_pw_reset_rate: dict = {}
_PW_RESET_MAX_REQUESTS = 3
_PW_RESET_WINDOW_SECONDS = 600  # 10 minutes

# Google Drive sync configuration
GDRIVE_SERVICE_ACCOUNT_FILE = os.environ.get('GDRIVE_SERVICE_ACCOUNT_FILE', './service_account.json')
GDRIVE_FOLDER_ID = os.environ.get('GDRIVE_FOLDER_ID', '')
GDRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
_gdrive_service_cache = None
_gdrive_service_cache_time = None
_GDRIVE_SERVICE_CACHE_TTL = 3600  # 1 hour


_SYNC_JOBS_COLLECTION = "sync_jobs"
_SYNC_SOURCE_ITEMS_COLLECTION = "sync_source_items"
_YOUTUBE_SUBSCRIPTIONS_COLLECTION = "youtube_subscriptions"
_SYNC_JOB_SOURCES = {"gdrive", "blogs", "youtube"}
_SYNC_ITEM_PENDING = "pending"
_SYNC_ITEM_PROCESSING = "processing"
_SYNC_ITEM_COMPLETED = "completed"
_SYNC_ITEM_FAILED = "failed"
_SYNC_JOB_PAUSED = "paused"
_SYNC_ITEM_STATUSES = {
    _SYNC_ITEM_PENDING,
    _SYNC_ITEM_PROCESSING,
    _SYNC_ITEM_COMPLETED,
    _SYNC_ITEM_FAILED,
}
_SYNC_PROCESSING_LEASE_SECONDS = 900
_SYNC_MAX_ATTEMPTS = 2
_SYNC_RETRY_BASE_SECONDS = 30
_SYNC_RETRY_MAX_SECONDS = 900
_SYNC_WORKER_POLL_SECONDS = 15
_EMPTY_INDEX_ERROR_PREFIX = "No saved index found in"
_SYNC_WORKER_INSTANCE_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
_sync_worker_task: Optional[asyncio.Task] = None
_sync_worker_wakeup: Optional[asyncio.Event] = None
_index_write_lock = threading.RLock()


def _get_gdrive_service():
    """Initialize authenticated Google Drive service with caching."""
    global _gdrive_service_cache, _gdrive_service_cache_time
    now = datetime.now(timezone.utc)

    if (_gdrive_service_cache is not None and
        _gdrive_service_cache_time is not None and
        (now - _gdrive_service_cache_time).total_seconds() < _GDRIVE_SERVICE_CACHE_TTL):
        return _gdrive_service_cache

    try:
        service_account_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT", "").strip()
        if service_account_json:
            credentials = service_account.Credentials.from_service_account_info(
                json.loads(service_account_json),
                scopes=GDRIVE_SCOPES,
            )
        else:
            credentials = service_account.Credentials.from_service_account_file(
                GDRIVE_SERVICE_ACCOUNT_FILE,
                scopes=GDRIVE_SCOPES,
            )
        service = build('drive', 'v3', credentials=credentials)
        _gdrive_service_cache = service
        _gdrive_service_cache_time = now
        logger.info("Google Drive service initialized and cached")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize Google Drive service: {e}")
        raise


def _get_gdrive_sync_state():
    """Get last Google Drive sync state from Firestore."""
    project = os.environ.get('GOOGLE_CLOUD_PROJECT', '')
    if not project:
        return None

    try:
        db_fs = _firestore_client()
        doc = db_fs.collection('system_config').document('gdrive_sync').get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        logger.warning(f"Failed to read Google Drive sync state: {e}")

    return None


def _save_gdrive_sync_state(last_sync_time: str):
    """Save Google Drive sync state to Firestore."""
    project = os.environ.get('GOOGLE_CLOUD_PROJECT', '')
    if not project:
        return

    db_fs = _firestore_client()
    db_fs.collection('system_config').document('gdrive_sync').set(
        {'last_sync_time': last_sync_time},
        merge=True
    )
    logger.info(f"Saved Google Drive sync state: {last_sync_time}")


def _download_gdrive_file(drive_service, file_id: str) -> bytes:
    """Download a file from Google Drive as bytes."""
    try:
        request = drive_service.files().get_media(fileId=file_id)
        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request, chunksize=1024*1024)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        return file_buffer.getvalue()
    except Exception as e:
        logger.error(f"Failed to download file {file_id}: {e}")
        raise


def _download_gdrive_file_to_path(drive_service, file_id: str, destination: Path) -> None:
    try:
        request = drive_service.files().get_media(fileId=file_id)
        with open(destination, "wb") as file_handle:
            downloader = MediaIoBaseDownload(file_handle, request, chunksize=1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
    except Exception as exc:
        logger.error("Failed to download Google Drive file %s: %s", file_id, exc)
        raise


def _sync_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sync_jobs_client():
    if not os.environ.get("GOOGLE_CLOUD_PROJECT", ""):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sync jobs require Firestore in production mode.",
        )
    return _firestore_client()


def _sync_source_url(source: str, source_id: str) -> str:
    if source == "gdrive":
        return f"gdrive://{source_id}"
    if source == "blogs":
        return f"cms://blog/{source_id}"
    if source == "youtube":
        return f"https://www.youtube.com/watch?v={source_id}"
    raise ValueError(f"Unsupported sync source: {source}")


def _sync_item_document_id(source: str, source_id: str) -> str:
    digest = hashlib.sha256(f"{source}:{source_id}".encode()).hexdigest()
    return f"{source}-{digest}"


def _sync_source_item_reference(
    db_fs,
    source: str,
    source_id: str,
    dedupe_key: Optional[str] = None,
):
    return db_fs.collection(_SYNC_SOURCE_ITEMS_COLLECTION).document(
        _sync_item_document_id(source, dedupe_key or source_id)
    )


def _parse_sync_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _create_sync_job(
    source: str,
    discovered_after: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> str:
    if source not in _SYNC_JOB_SOURCES:
        raise ValueError(f"Unsupported sync source: {source}")

    job_id = uuid.uuid4().hex
    now = _sync_timestamp()
    _sync_jobs_client().collection(_SYNC_JOBS_COLLECTION).document(job_id).set({
        "source": source,
        "status": "discovering",
        "discovered_after": discovered_after,
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
        "metadata": metadata or {},
    })
    return job_id


def _add_sync_job_items(job_id: str, source: str, items: list[dict]) -> int:
    if source not in _SYNC_JOB_SOURCES:
        raise ValueError(f"Unsupported sync source: {source}")

    db_fs = _sync_jobs_client()
    job_ref = db_fs.collection(_SYNC_JOBS_COLLECTION).document(job_id)
    item_collection = job_ref.collection("items")
    added = 0

    for item in items:
        source_id = str(item.get("source_id", "")).strip()
        if not source_id:
            raise ValueError("Each sync job item requires a source_id.")
        dedupe_key = str(item.get("dedupe_key") or source_id).strip()
        if not dedupe_key:
            raise ValueError("Each sync job item requires a non-empty dedupe_key.")

        item_ref = item_collection.document(_sync_item_document_id(source, dedupe_key))
        source_ref = _sync_source_item_reference(db_fs, source, source_id, dedupe_key)
        if source_ref.get().exists:
            continue

        now = _sync_timestamp()
        item_data = {
            "source": source,
            "source_id": source_id,
            "dedupe_key": dedupe_key,
            "source_url": _sync_source_url(source, source_id),
            "name": item.get("name", source_id),
            "created_time": item.get("created_time"),
            "status": _SYNC_ITEM_PENDING,
            "attempts": 0,
            "next_attempt_at": None,
            "lease_expires_at": None,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "metadata": item.get("metadata", {}),
            "created_at": now,
            "updated_at": now,
        }
        batch = db_fs.batch()
        batch.set(item_ref, item_data)
        batch.set(source_ref, {
            "source": source,
            "source_id": source_id,
            "dedupe_key": dedupe_key,
            "job_id": job_id,
            "item_id": item_ref.id,
            "status": _SYNC_ITEM_PENDING,
            "created_time": item.get("created_time"),
            "created_at": now,
            "updated_at": now,
        })
        batch.commit()
        added += 1

    has_items = any(item_collection.limit(1).stream())
    job_ref.set({
        "status": "queued" if has_items else "completed",
        "updated_at": _sync_timestamp(),
        "completed_at": _sync_timestamp() if not has_items else None,
    }, merge=True)
    return added


def _claim_next_sync_item(job_id: str) -> Optional[dict]:
    from google.cloud import firestore

    db_fs = _sync_jobs_client()
    job_ref = db_fs.collection(_SYNC_JOBS_COLLECTION).document(job_id)
    for candidate in job_ref.collection("items").order_by("created_at").stream():
        candidate_item = candidate.to_dict()
        if candidate_item.get("status") != _SYNC_ITEM_PENDING:
            continue
        claim_token = uuid.uuid4().hex

        @firestore.transactional
        def claim_item(transaction):
            job_snapshot = job_ref.get(transaction=transaction)
            job = job_snapshot.to_dict() if job_snapshot.exists else None
            if not job or job.get("status") not in {"queued", "running", "discovering"}:
                return None

            item_snapshot = candidate.reference.get(transaction=transaction)
            if not item_snapshot.exists:
                return None
            item = item_snapshot.to_dict() or {}
            if item.get("status") != _SYNC_ITEM_PENDING:
                return None

            now = datetime.now(timezone.utc)
            next_attempt_at = item.get("next_attempt_at")
            if next_attempt_at:
                try:
                    next_attempt = _parse_sync_datetime(next_attempt_at)
                except ValueError:
                    next_attempt = now
                if next_attempt > now:
                    return None

            attempts = int(item.get("attempts", 0)) + 1
            now_text = now.isoformat()
            transaction.update(candidate.reference, {
                "status": _SYNC_ITEM_PROCESSING,
                "attempts": attempts,
                "started_at": now_text,
                "next_attempt_at": None,
                "lease_expires_at": (now + timedelta(seconds=_SYNC_PROCESSING_LEASE_SECONDS)).isoformat(),
                "updated_at": now_text,
                "error": None,
                "worker_id": _SYNC_WORKER_INSTANCE_ID,
                "claim_token": claim_token,
            })
            source_ref = _sync_source_item_reference(
                db_fs, item["source"], item["source_id"], item.get("dedupe_key")
            )
            transaction.set(source_ref, {
                "status": _SYNC_ITEM_PROCESSING,
                "attempts": attempts,
                "worker_id": _SYNC_WORKER_INSTANCE_ID,
                "updated_at": now_text,
            }, merge=True)
            item["id"] = candidate.id
            item["attempts"] = attempts
            item["worker_id"] = _SYNC_WORKER_INSTANCE_ID
            item["claim_token"] = claim_token
            return item

        item = claim_item(db_fs.transaction())
        if item:
            return item
    return None


def _complete_sync_item(
    job_id: str,
    item_id: str,
    chunks_stored: int,
    claim_token: Optional[str],
) -> bool:
    from google.cloud import firestore

    now = _sync_timestamp()
    db_fs = _sync_jobs_client()
    item_ref = db_fs.collection(_SYNC_JOBS_COLLECTION).document(job_id).collection("items").document(item_id)

    @firestore.transactional
    def complete_item(transaction):
        item_snapshot = item_ref.get(transaction=transaction)
        item = item_snapshot.to_dict() if item_snapshot.exists else None
        if not item or item.get("status") != _SYNC_ITEM_PROCESSING:
            return False
        if item.get("claim_token") != claim_token:
            return False

        transaction.update(item_ref, {
            "status": _SYNC_ITEM_COMPLETED,
            "chunks_stored": chunks_stored,
            "completed_at": now,
            "lease_expires_at": None,
            "claim_token": None,
            "updated_at": now,
            "error": None,
        })
        source_ref = _sync_source_item_reference(
            db_fs, item["source"], item["source_id"], item.get("dedupe_key")
        )
        transaction.set(source_ref, {
            "status": _SYNC_ITEM_COMPLETED,
            "chunks_stored": chunks_stored,
            "completed_at": now,
            "updated_at": now,
            "error": None,
        }, merge=True)
        return True

    return complete_item(db_fs.transaction())


def _fail_sync_item(
    job_id: str,
    item_id: str,
    error: str,
    retryable: bool = True,
    claim_token: Optional[str] = None,
) -> bool:
    from google.cloud import firestore

    db_fs = _sync_jobs_client()
    item_ref = db_fs.collection(_SYNC_JOBS_COLLECTION).document(job_id).collection("items").document(item_id)

    @firestore.transactional
    def fail_item(transaction):
        item_snapshot = item_ref.get(transaction=transaction)
        item = item_snapshot.to_dict() if item_snapshot.exists else None
        if not item or item.get("status") != _SYNC_ITEM_PROCESSING:
            return False
        if item.get("claim_token") != claim_token:
            return False

        attempts = int(item.get("attempts", 0))
        next_status = _SYNC_ITEM_PENDING if retryable and attempts < _SYNC_MAX_ATTEMPTS else _SYNC_ITEM_FAILED
        retry_delay = min(
            _SYNC_RETRY_BASE_SECONDS * (2 ** max(attempts - 1, 0)),
            _SYNC_RETRY_MAX_SECONDS,
        )
        next_attempt_at = (
            (datetime.now(timezone.utc) + timedelta(seconds=retry_delay)).isoformat()
            if next_status == _SYNC_ITEM_PENDING else None
        )
        now = _sync_timestamp()
        transaction.update(item_ref, {
            "status": next_status,
            "lease_expires_at": None,
            "next_attempt_at": next_attempt_at,
            "claim_token": None,
            "updated_at": now,
            "error": error[:2000],
        })
        source_ref = _sync_source_item_reference(
            db_fs, item["source"], item["source_id"], item.get("dedupe_key")
        )
        transaction.set(source_ref, {
            "status": next_status,
            "attempts": attempts,
            "updated_at": now,
            "error": error[:2000],
        }, merge=True)
        return True

    return fail_item(db_fs.transaction())


def _recover_stale_sync_items(job_id: str) -> int:
    db_fs = _sync_jobs_client()
    stale_before = datetime.now(timezone.utc)
    recovered = 0
    for item_doc in db_fs.collection(_SYNC_JOBS_COLLECTION).document(job_id).collection("items").where(
        "status", "==", _SYNC_ITEM_PROCESSING
    ).stream():
        item = item_doc.to_dict()
        lease_expires_at = item.get("lease_expires_at")
        if not lease_expires_at:
            continue
        try:
            lease_expiry = datetime.fromisoformat(lease_expires_at.replace("Z", "+00:00"))
        except ValueError:
            lease_expiry = stale_before
        if lease_expiry > stale_before:
            continue
        _fail_sync_item(
            job_id,
            item_doc.id,
            "Processing lease expired after an interrupted worker.",
            claim_token=item.get("claim_token"),
        )
        recovered += 1
    return recovered


def _recover_empty_index_failures() -> int:
    db_fs = _sync_jobs_client()
    recovered = 0
    for job_doc in db_fs.collection(_SYNC_JOBS_COLLECTION).stream():
        job_ref = job_doc.reference
        for item_doc in job_ref.collection("items").stream():
            item = item_doc.to_dict() or {}
            if item.get("status") != _SYNC_ITEM_FAILED:
                continue
            if not str(item.get("error") or "").startswith(_EMPTY_INDEX_ERROR_PREFIX):
                continue

            now = _sync_timestamp()
            item_doc.reference.update({
                "status": _SYNC_ITEM_PENDING,
                "attempts": 0,
                "next_attempt_at": None,
                "lease_expires_at": None,
                "started_at": None,
                "completed_at": None,
                "updated_at": now,
                "error": None,
                "worker_id": None,
                "claim_token": None,
            })
            _sync_source_item_reference(
                db_fs, item["source"], item["source_id"], item.get("dedupe_key")
            ).set({
                "status": _SYNC_ITEM_PENDING,
                "attempts": 0,
                "updated_at": now,
                "error": None,
            }, merge=True)
            job_ref.set({
                "status": "queued",
                "completed_at": None,
                "updated_at": now,
            }, merge=True)
            recovered += 1
    return recovered


def _pause_sync_job(job_id: str) -> dict:
    db_fs = _sync_jobs_client()
    job_ref = db_fs.collection(_SYNC_JOBS_COLLECTION).document(job_id)
    job_doc = job_ref.get()
    if not job_doc.exists:
        raise HTTPException(status_code=404, detail="Sync job not found.")

    current_status = (job_doc.to_dict() or {}).get("status")
    if current_status == "completed":
        raise HTTPException(status_code=409, detail="Completed jobs cannot be paused.")

    now = _sync_timestamp()
    job_ref.set({
        "status": _SYNC_JOB_PAUSED,
        "paused_at": now,
        "updated_at": now,
    }, merge=True)
    reset_items = 0
    for item_doc in job_ref.collection("items").stream():
        item = item_doc.to_dict() or {}
        if item.get("status") == _SYNC_ITEM_COMPLETED:
            continue
        item_doc.reference.update({
            "status": _SYNC_ITEM_PENDING,
            "attempts": 0,
            "next_attempt_at": None,
            "lease_expires_at": None,
            "started_at": None,
            "completed_at": None,
            "updated_at": now,
            "error": None,
            "worker_id": None,
            "claim_token": None,
        })
        _sync_source_item_reference(
            db_fs, item["source"], item["source_id"], item.get("dedupe_key")
        ).set({
            "status": _SYNC_ITEM_PENDING,
            "attempts": 0,
            "updated_at": now,
            "error": None,
        }, merge=True)
        reset_items += 1
    job_ref.set({"paused_reset_items": reset_items}, merge=True)
    return _sync_job_status(job_id)


def _restart_sync_job(job_id: str) -> dict:
    db_fs = _sync_jobs_client()
    job_ref = db_fs.collection(_SYNC_JOBS_COLLECTION).document(job_id)
    job_doc = job_ref.get()
    if not job_doc.exists:
        raise HTTPException(status_code=404, detail="Sync job not found.")

    job = job_doc.to_dict() or {}
    if job.get("status") != _SYNC_JOB_PAUSED:
        raise HTTPException(status_code=409, detail="Pause the sync job before restarting it.")

    now = _sync_timestamp()
    job_ref.set({
        "status": "queued",
        "completed_at": None,
        "paused_at": None,
        "restarted_at": now,
        "updated_at": now,
    }, merge=True)
    _wake_sync_worker()
    return _sync_job_status(job_id)


def _remove_gdrive_video_items(job_id: str) -> dict:
    db_fs = _sync_jobs_client()
    job_ref = db_fs.collection(_SYNC_JOBS_COLLECTION).document(job_id)
    job_doc = job_ref.get()
    if not job_doc.exists:
        raise HTTPException(status_code=404, detail="Sync job not found.")

    job = job_doc.to_dict() or {}
    if job.get("source") != "gdrive":
        raise HTTPException(status_code=400, detail="Video removal is only available for Google Drive jobs.")

    items = list(job_ref.collection("items").stream())
    processing_items = [
        item_doc.id
        for item_doc in items
        if (item_doc.to_dict() or {}).get("status") == _SYNC_ITEM_PROCESSING
    ]
    if processing_items:
        raise HTTPException(
            status_code=409,
            detail="Wait for the active item to finish before removing videos.",
        )

    batch = db_fs.batch()
    removed = 0
    for item_doc in items:
        item = item_doc.to_dict() or {}
        mime_type = str((item.get("metadata") or {}).get("mime_type") or "")
        if not mime_type.startswith("video/"):
            continue
        batch.delete(item_doc.reference)
        batch.delete(_sync_source_item_reference(
            db_fs, item["source"], item["source_id"], item.get("dedupe_key")
        ))
        removed += 1

    if removed:
        batch.commit()
        _refresh_sync_job_status(job_id)

    result = _sync_job_status(job_id)
    result["removed_video_items"] = removed
    return result


def _remove_sync_job_item(job_id: str, item_id: str) -> dict:
    db_fs = _sync_jobs_client()
    job_ref = db_fs.collection(_SYNC_JOBS_COLLECTION).document(job_id)
    job_doc = job_ref.get()
    if not job_doc.exists:
        raise HTTPException(status_code=404, detail="Sync job not found.")

    item_ref = job_ref.collection("items").document(item_id)
    item_doc = item_ref.get()
    if not item_doc.exists:
        raise HTTPException(status_code=404, detail="Sync job item not found.")

    item = item_doc.to_dict() or {}
    if item.get("status") == _SYNC_ITEM_PROCESSING:
        raise HTTPException(
            status_code=409,
            detail="A processing item cannot be removed. Pause the job and wait for it to stop first.",
        )
    if item.get("status") == _SYNC_ITEM_COMPLETED:
        raise HTTPException(
            status_code=409,
            detail="Completed items are already indexed and cannot be removed from the queue.",
        )

    batch = db_fs.batch()
    batch.delete(item_ref)
    batch.delete(_sync_source_item_reference(
        db_fs, item["source"], item["source_id"], item.get("dedupe_key")
    ))
    batch.commit()
    _refresh_sync_job_status(job_id)

    result = _sync_job_status(job_id)
    result["removed_item"] = {"id": item_id, "name": item.get("name")}
    return result


def _refresh_sync_job_status(job_id: str) -> None:
    job_ref = _sync_jobs_client().collection(_SYNC_JOBS_COLLECTION).document(job_id)
    job = job_ref.get().to_dict() or {}
    if job.get("status") == _SYNC_JOB_PAUSED:
        return
    statuses = [
        item_doc.to_dict().get("status")
        for item_doc in job_ref.collection("items").stream()
    ]
    if not statuses:
        job_ref.set({
            "status": "completed",
            "completed_at": _sync_timestamp(),
            "updated_at": _sync_timestamp(),
        }, merge=True)
        return

    if _SYNC_ITEM_PENDING in statuses or _SYNC_ITEM_PROCESSING in statuses:
        job_ref.set({"status": "running", "updated_at": _sync_timestamp()}, merge=True)
        return

    job_ref.set({
        "status": "completed" if _SYNC_ITEM_FAILED not in statuses else "completed_with_failures",
        "completed_at": _sync_timestamp(),
        "updated_at": _sync_timestamp(),
    }, merge=True)


def _claim_next_sync_work() -> Optional[tuple[str, dict]]:
    db_fs = _sync_jobs_client()
    for job_doc in db_fs.collection(_SYNC_JOBS_COLLECTION).order_by("created_at").stream():
        job = job_doc.to_dict()
        job_status = job.get("status")
        if job_status not in {"queued", "running", "discovering", _SYNC_JOB_PAUSED}:
            continue
        _recover_stale_sync_items(job_doc.id)
        if job_status == _SYNC_JOB_PAUSED:
            continue
        item = _claim_next_sync_item(job_doc.id)
        if item:
            job_doc.reference.set({"status": "running", "updated_at": _sync_timestamp()}, merge=True)
            return job_doc.id, item
        _refresh_sync_job_status(job_doc.id)
    return None


def _run_index_mutation(operation):
    with _index_write_lock:
        return operation()


def _serialized_index_mutation(handler):
    if inspect.iscoroutinefunction(handler):
        @functools.wraps(handler)
        async def async_wrapped(*args, **kwargs):
            with _index_write_lock:
                return await handler(*args, **kwargs)
        return async_wrapped

    @functools.wraps(handler)
    def wrapped(*args, **kwargs):
        return _run_index_mutation(lambda: handler(*args, **kwargs))
    return wrapped


def _process_gdrive_sync_item(item: dict) -> int:
    file_id = item["source_id"]
    file_name = Path(item.get("name") or file_id).name
    if not file_name or file_name == ".":
        raise ValueError("Google Drive sync item has no usable filename.")

    drive_service = _get_gdrive_service()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / file_name
        _download_gdrive_file_to_path(drive_service, file_id, local_path)

        def ingest() -> int:
            rag = _get_rag()
            chunks_stored = rag.ingest_file(
                str(local_path),
                section="gdrive",
                source_url=item["source_url"],
            )
            if chunks_stored <= 0:
                raise RuntimeError("No indexable content was extracted from the Google Drive file.")
            _save_and_sync(rag)
            return chunks_stored

        return _run_index_mutation(ingest)


def _process_blog_sync_item(item: dict) -> int:
    metadata = item.get("metadata", {})
    title = metadata.get("title") or item.get("name") or item["source_id"]
    pdf_url = (metadata.get("pdf_url") or "").strip()

    def ingest_text() -> int:
        content = (metadata.get("content") or "").strip()
        if not content:
            raise RuntimeError("Blog item has no indexable text content.")
        rag = _get_rag()
        chunks_stored = rag.ingest_raw_documents([{
            "title": title,
            "content": content,
            "section": "blogs",
            "url": item["source_url"],
        }])
        if chunks_stored <= 0:
            raise RuntimeError("No indexable content was extracted from the blog entry.")
        _save_and_sync(rag)
        return chunks_stored

    if not pdf_url:
        return _run_index_mutation(ingest_text)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / f"{item['source_id']}.pdf"
        response = _req.get(pdf_url, stream=True, timeout=60)
        response.raise_for_status()
        with open(local_path, "wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_handle.write(chunk)

        def ingest_pdf() -> int:
            rag = _get_rag()
            chunks_stored = rag.ingest_file(
                str(local_path),
                section="blogs",
                source_url=item["source_url"],
            )
            if chunks_stored <= 0:
                raise RuntimeError("No indexable content was extracted from the blog PDF.")
            _save_and_sync(rag)
            return chunks_stored

        return _run_index_mutation(ingest_pdf)


def _process_youtube_sync_item(item: dict) -> int:
    video_url = item["source_url"]

    def ingest() -> int:
        rag = _get_rag()
        chunks_stored = rag.ingest_videos([video_url], section="video")
        if chunks_stored <= 0:
            raise RuntimeError("No indexable content was extracted from the YouTube video.")
        _save_and_sync(rag)
        return chunks_stored

    return _run_index_mutation(ingest)


def _process_sync_item(item: dict) -> int:
    if item.get("source") == "gdrive":
        return _process_gdrive_sync_item(item)
    if item.get("source") == "blogs":
        return _process_blog_sync_item(item)
    if item.get("source") == "youtube":
        return _process_youtube_sync_item(item)
    raise RuntimeError(f"No worker handler is registered for sync source {item.get('source')!r}.")


def _wake_sync_worker() -> None:
    if _sync_worker_wakeup is not None:
        _sync_worker_wakeup.set()


async def _wait_for_sync_worker_wakeup() -> None:
    if _sync_worker_wakeup is None:
        await asyncio.sleep(_SYNC_WORKER_POLL_SECONDS)
        return
    try:
        await asyncio.wait_for(_sync_worker_wakeup.wait(), timeout=_SYNC_WORKER_POLL_SECONDS)
    except asyncio.TimeoutError:
        pass


async def _sync_worker_loop() -> None:
    logger.info("Sync worker started.")
    while True:
        try:
            if _sync_worker_wakeup is not None:
                _sync_worker_wakeup.clear()

            if not os.environ.get("GOOGLE_CLOUD_PROJECT", ""):
                await _wait_for_sync_worker_wakeup()
                continue

            recovered = await asyncio.to_thread(_recover_empty_index_failures)
            if recovered:
                logger.warning("Worker: requeued %d item(s) after empty-index recovery.", recovered)

            work = await asyncio.to_thread(_claim_next_sync_work)
            if not work:
                await _wait_for_sync_worker_wakeup()
                continue

            job_id, item = work
            try:
                chunks_stored = await asyncio.to_thread(_process_sync_item, item)
                completed = await asyncio.to_thread(
                    _complete_sync_item,
                    job_id,
                    item["id"],
                    chunks_stored,
                    item.get("claim_token"),
                )
                if not completed:
                    logger.warning("Worker lost ownership before completing job=%s item=%s", job_id, item["id"])
            except Exception as exc:
                logger.exception("Sync worker failed job=%s item=%s", job_id, item["id"])
                failed = await asyncio.to_thread(
                    _fail_sync_item,
                    job_id,
                    item["id"],
                    str(exc),
                    True,
                    item.get("claim_token"),
                )
                if not failed:
                    logger.warning("Worker lost ownership before failing job=%s item=%s", job_id, item["id"])
            finally:
                await asyncio.to_thread(_refresh_sync_job_status, job_id)
        except asyncio.CancelledError:
            logger.info("Sync worker stopped.")
            raise
        except Exception as exc:
            logger.exception("Sync worker loop error: %s", exc)
            await _wait_for_sync_worker_wakeup()


def _start_sync_worker() -> None:
    global _sync_worker_task, _sync_worker_wakeup
    if not os.environ.get("GOOGLE_CLOUD_PROJECT", ""):
        return
    if _sync_worker_task is not None and not _sync_worker_task.done():
        return
    _sync_worker_wakeup = asyncio.Event()
    _sync_worker_task = asyncio.create_task(_sync_worker_loop(), name="sync-worker")


async def _stop_sync_worker() -> None:
    global _sync_worker_task, _sync_worker_wakeup
    if _sync_worker_task is None:
        return
    _sync_worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await _sync_worker_task
    _sync_worker_task = None
    _sync_worker_wakeup = None


def _sync_job_status(job_id: str, item_limit: int = 100, item_offset: int = 0) -> dict:
    db_fs = _sync_jobs_client()
    job_doc = db_fs.collection(_SYNC_JOBS_COLLECTION).document(job_id).get()
    if not job_doc.exists:
        raise HTTPException(status_code=404, detail="Sync job not found.")

    counts = {item_status: 0 for item_status in _SYNC_ITEM_STATUSES}
    items = []
    for item_index, item_doc in enumerate(
        job_doc.reference.collection("items").order_by("created_at").stream()
    ):
        item = item_doc.to_dict()
        item["id"] = item_doc.id
        if item_offset <= item_index < item_offset + item_limit:
            items.append(item)
        item_status = item.get("status")
        if item_status in counts:
            counts[item_status] += 1

    job = job_doc.to_dict()
    job["id"] = job_doc.id
    job["counts"] = counts
    job["items"] = items
    job["item_offset"] = item_offset
    job["next_item_offset"] = (
        item_offset + item_limit
        if item_offset + item_limit < sum(counts.values()) else None
    )
    return job


def _list_sync_jobs(limit: int = 20) -> list[dict]:
    from google.cloud.firestore import Query as FirestoreQuery

    db_fs = _sync_jobs_client()
    jobs = []
    for job_doc in db_fs.collection(_SYNC_JOBS_COLLECTION).order_by(
        "created_at", direction=FirestoreQuery.DESCENDING
    ).limit(limit).stream():
        job = job_doc.to_dict()
        job["id"] = job_doc.id
        jobs.append(job)
    return jobs


def _load_model_from_firestore() -> Optional[str]:
    """Load saved model from Firestore, or None if not found/not configured."""
    if not os.environ.get("GOOGLE_CLOUD_PROJECT", ""):
        return None
    try:
        db = _firestore_client()
        doc = db.collection("system_config").document("admin").get()
        if doc.exists:
            return doc.to_dict().get("model")
    except Exception as exc:
        logger.warning("Failed to load model from Firestore: %s", exc)
    return None


def _save_model_to_firestore(model: str) -> None:
    """Save selected model to Firestore."""
    if not os.environ.get("GOOGLE_CLOUD_PROJECT", ""):
        return
    try:
        db = _firestore_client()
        db.collection("system_config").document("admin").set(
            {"model": model}, merge=True
        )
    except Exception as exc:
        logger.error("Failed to save model to Firestore: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Firestore write failed: {exc}")


def _fetch_anthropic_models() -> list[dict]:
    """Fetch available Anthropic models and filter for chat models. Caches for 5 min."""
    now = datetime.now(timezone.utc)

    # Return cached models if still valid
    if (_model_cache["models"] is not None and
        _model_cache["timestamp"] is not None and
        (now - _model_cache["timestamp"]).total_seconds() < _MODEL_CACHE_TTL):
        return _model_cache["models"]

    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set — cannot fetch models")
            return []

        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

        page = client.models.list()
        all_models = page.data

        # Filter for chat models (type == "model" and not embedding models)
        chat_models = [
            {"id": m.id, "display_name": getattr(m, "display_name", m.id)}
            for m in all_models
            if getattr(m, "type", None) == "model" and "embed" not in m.id.lower()
        ]

        # Sort by id for consistent ordering
        chat_models.sort(key=lambda x: x["id"])

        # Cache the result
        _model_cache["models"] = chat_models
        _model_cache["timestamp"] = now

        return chat_models
    except Exception as exc:
        logger.error("Failed to fetch Anthropic models: %s", exc)
        return []


def _get_rag() -> RAGOrchestrator:
    global _rag
    if _rag is None:
        _rag = RAGOrchestrator(
            gemini_api_key=_current_config["gemini_api_key"],
            youtube_api_key=_current_config["youtube_api_key"],
            hf_model_name=_current_config["hf_model_name"],
            chunk_size=_current_config["chunk_size"],
            chunk_overlap=_current_config["chunk_overlap"],
            dedup_threshold=_current_config["dedup_threshold"],
            min_tokens=_current_config["min_tokens"],
            index_dir=_current_config["index_dir"],
            cache_dir=_current_config["cache_dir"],
            follow_external=_current_config["follow_external"],
            device=_current_config["device"],
        )
    return _rag


def _get_gemini_rag(
    system_prompt: Optional[str] = None,
) -> RAG:
    return RAG(
        db=_get_rag().db,
        gemini_api_key=_current_config["gemini_api_key"],
        anthropic_api_key=_current_config["anthropic_api_key"],
        model=_current_config["model"],
        top_k=_current_config["top_k"],
        system_prompt=system_prompt,
        session_store=_session_store,
    )


def _save_and_sync(rag: RAGOrchestrator) -> None:
    """Save index to disk then push to GCS (if GCS_BUCKET is configured)."""
    rag.save()
    bucket = _current_config.get("gcs_bucket", "")
    if bucket:
        _upload_index_to_gcs(bucket, _current_config["index_dir"])


def _scan_cleanup_candidates(db, req: CleanupRequest) -> tuple[list[int], list[dict]]:
    """
    Scan all FAISS chunks and return (internal_ids_to_delete, sample_dicts).
    Applies short-chunk, repeated-word, and regex filters per the request.
    """
    import re as _re
    import collections as _col

    compiled: list = []
    if req.regex_enabled:
        for p in req.regex_patterns:
            try:
                compiled.append(_re.compile(p, _re.IGNORECASE))
            except Exception:
                pass

    flagged: list[int] = []
    samples: list[dict] = []

    for iid, chunk in list(db._meta.items()):
        if req.section_filter and chunk.section != req.section_filter:
            continue

        content = "\n".join(filter(None, [
            getattr(chunk, "text", None),
            getattr(chunk, "raw_content", None),
        ])).strip()
        words = content.split()
        total = len(words)
        reason: Optional[str] = None

        if req.short_chunk_enabled:
            if total < req.short_min_tokens or len(content) < req.short_min_chars:
                reason = f"short ({total} tokens)"

        if reason is None and req.repeated_word_enabled and total > 0:
            long_words = [w.lower() for w in words if len(w) >= req.repeated_word_min_length]
            counts = _col.Counter(long_words)
            if any(c >= req.repeated_word_min_count for c in counts.values()):
                top = counts.most_common(1)[0]
                reason = f"repeated '{top[0]}' x{top[1]}"

        if reason is None and req.regex_enabled and compiled:
            for pat in compiled:
                if pat.search(content):
                    reason = f"regex: {pat.pattern}"
                    break

        if reason:
            flagged.append(iid)
            if len(samples) < 20:
                samples.append({
                    "doc_title":   getattr(chunk, "doc_title", ""),
                    "section":     getattr(chunk, "section", ""),
                    "chunk_index": getattr(chunk, "chunk_index", 0),
                    "doc_type":    getattr(chunk, "doc_type", ""),
                    "preview":     content[:300],
                    "reason":      reason,
                })

    return flagged, samples


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ConfigUpdate(BaseModel):
    gemini_api_key:    Optional[str]   = None
    anthropic_api_key: Optional[str]   = None
    youtube_api_key:   Optional[str]   = None
    hf_model_name:     Optional[str]   = None
    chunk_size:        Optional[int]   = None
    chunk_overlap:     Optional[int]   = None
    dedup_threshold:   Optional[float] = None
    min_tokens:        Optional[int]   = None
    index_dir:         Optional[str]   = None
    cache_dir:         Optional[str]   = None
    follow_external:   Optional[bool]  = None
    device:            Optional[str]   = None
    model:             Optional[str]   = None
    top_k:             Optional[int]   = None
    gcs_bucket:        Optional[str]   = None


class SetupRequest(BaseModel):
    password: str


class LoginRequest(BaseModel):
    password: str


class SetModelRequest(BaseModel):
    model: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class IngestRequest(BaseModel):
    url:     str
    rebuild: bool = False


class QueryRequest(BaseModel):
    question:        str
    top_k:           int           = Field(default=6, ge=1, le=20)
    section_filter:  Optional[str] = None
    doc_type_filter: Optional[str] = None
    system_prompt:   Optional[str] = None
    model:           Optional[str] = None
    session_id:      Optional[str] = None


class FolderIngestRequest(BaseModel):
    folder_path: str
    section:     str  = "general"
    recursive:   bool = True


class RawDocumentItem(BaseModel):
    title:    str = "Untitled"
    content:  str
    section:  str = "general"
    url:      str = ""
    doc_type: str = "text"


class RawDocumentsRequest(BaseModel):
    documents: list[RawDocumentItem]


class VideosIngestRequest(BaseModel):
    urls:    list[str]
    section: str = "video"


class CleanupRequest(BaseModel):
    repeated_word_enabled:    bool        = False
    repeated_word_min_length: int         = 4
    repeated_word_min_count:  int         = 10
    repeated_word_window:     int         = 0      # reserved, kept for schema compat
    short_chunk_enabled:      bool        = False
    short_min_tokens:         int         = 20
    short_min_chars:          int         = 30
    regex_enabled:            bool        = False
    regex_patterns:           list[str]   = []
    section_filter:           Optional[str] = None


# ---------------------------------------------------------------------------
# Thread-pool helper
# ---------------------------------------------------------------------------

def _run_llm_in_thread(
    g: RAG,
    question: str,
    top_k: int,
    section_filter: Optional[str],
    doc_type_filter: Optional[str],
    session_id: Optional[str],
    token_queue: "_sync_queue.Queue[tuple[str, object]]",
) -> None:
    """
    Runs stream_answer() synchronously in a worker thread.
    Puts items into token_queue:
      ('chunk', dict)         — one retrieved chunk (from a tool call)
      ('token', str)          — one text token
      ('done',  GeminiAnswer) — generator exhausted normally
      ('error', str)          — exception message
    """
    chunk_rank = [0]

    def on_chunks(results: list[dict]) -> None:
        for r in results:
            chunk_rank[0] += 1
            payload = {
                "rank":        chunk_rank[0],
                "score":       r["score"],
                "doc_index":   r["doc_index"],
                "doc_title":   r["doc_title"],
                "section":     r["section"],
                "doc_type":    r["doc_type"],
                "doc_url":     r["doc_url"],
                "chunk_index": r["chunk_index"],
                "raw_content": r["raw_content"],
            }
            token_queue.put(("chunk", payload))

    try:
        gen = g.stream_answer(
            question=question,
            top_k=top_k,
            section_filter=section_filter,
            doc_type_filter=doc_type_filter,
            session_id=session_id,
            on_chunks=on_chunks,
        )
        while True:
            try:
                token = next(gen)
                token_queue.put(("token", token))
            except StopIteration as e:
                token_queue.put(("done", e.value))
                return
    except Exception as exc:
        logger.error("LLM thread error: %s", exc)
        token_queue.put(("error", str(exc)))


# ---------------------------------------------------------------------------
# Public routes  (no auth required)
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """
    On container start:
    1. Load admin password hash from DB.
    2. Load model selection from Firestore.
    3. Download the FAISS index from GCS (if configured).
    """
    global _cached_admin_hash, _current_config

    # --- 1. Password Setup ---
    try:
        _cached_admin_hash = _get_admin_hash_from_db()
        if _cached_admin_hash:
            logger.info("Startup: admin password hash loaded from Firestore.")
        else:
            logger.info("Startup: no admin password set yet.")
    except Exception as exc:
        logger.error(
            "Startup: Firestore read failed — server will start but admin auth "
            "will return 503 until DB is reachable. Error: %s", exc
        )

    # --- 1b. Load Model ---
    try:
        saved_model = _load_model_from_firestore()
        if saved_model and saved_model.strip():
            _current_config["model"] = saved_model.strip()
            logger.info("Startup: model loaded from Firestore: %s", saved_model)
        else:
            logger.info("Startup: using default model: %s", _current_config["model"])
    except Exception as exc:
        logger.warning("Startup: failed to load model from Firestore, using default: %s", exc)

    # Validate that we have a model configured
    if not _current_config.get("model"):
        logger.error("Startup: no model configured, falling back to claude-sonnet-4-6")
        _current_config["model"] = "claude-sonnet-4-6"

    # --- 1c. Load System Prompt ---
    try:
        from rag_query import load_system_prompt
        load_system_prompt()
        logger.info("Startup: system prompt loaded.")
    except Exception as exc:
        logger.warning("Startup: failed to load system prompt, using default: %s", exc)

    # --- 2. GCS FAISS Download ---
    bucket = _current_config.get("gcs_bucket", "")
    if bucket:
        index_dir = _current_config["index_dir"]
        logger.info("Startup: downloading FAISS index from GCS bucket %s ...", bucket)
        ok = _download_index_from_gcs(bucket, index_dir)
        if ok:
            logger.info("Startup: FAISS index ready from GCS.")
        else:
            logger.warning("Startup: GCS download incomplete — starting with empty index.")
    else:
        logger.info("Startup: GCS_BUCKET not set — using local index (local dev mode).")

    # --- 2b. GCS System Prompt Download ---
    if bucket:
        logger.info("Startup: downloading system prompt from GCS bucket %s ...", bucket)
        ok = _download_system_prompt_from_gcs(bucket, "system_config")
        if not ok:
            logger.warning("Startup: GCS system prompt download failed — using default.")
        from rag_query import load_system_prompt
        load_system_prompt()
    else:
        logger.info("Startup: GCS_BUCKET not set — using local system prompt (local dev mode).")

    # --- 3. Warm up RAG (load FAISS index + rebuild BM25 before first request) ---
    try:
        _get_rag()
        logger.info("Startup: RAG index loaded and ready.")
    except Exception as exc:
        logger.error("Startup: RAG warm-up failed — first query will trigger lazy load. Error: %s", exc)

    try:
        recovered = await asyncio.to_thread(_recover_empty_index_failures)
        if recovered:
            logger.warning("Startup: requeued %d item(s) after empty-index recovery.", recovered)
    except Exception as exc:
        logger.error("Startup: empty-index recovery failed: %s", exc)

    _start_sync_worker()


@app.on_event("shutdown")
async def shutdown_event():
    await _stop_sync_worker()


@app.get("/")
def root():
    """
    API root — returns service info.
    Both frontends (chat UI + admin UI) are deployed separately on Vercel
    and call this service via its Cloud Run URL directly.
    """
    return {
        "service": "Portfolio RAG API",
        "version": "3.0",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return _get_rag().stats()


# ---------------------------------------------------------------------------
# Query — streaming SSE  (public)
# ---------------------------------------------------------------------------

@app.get("/query/stream")
async def query_stream(
    question:        str,
    top_k:           int  = 6,
    section_filter:  Optional[str] = None,
    doc_type_filter: Optional[str] = None,
    system_prompt:   Optional[str] = None,
    session_id:      Optional[str] = None,
):
    """
    SSE streaming endpoint.
    Events: chunk | token | done | error | ping
    """
    g = _get_gemini_rag(system_prompt=system_prompt)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            token_queue: _sync_queue.Queue = _sync_queue.Queue()
            loop = asyncio.get_event_loop()

            future = loop.run_in_executor(
                _thread_pool,
                _run_llm_in_thread,
                g, question, top_k, section_filter, doc_type_filter,
                session_id, token_queue,
            )

            ping_counter = 0
            final_answer = None

            while True:
                try:
                    kind, value = token_queue.get_nowait()
                except _sync_queue.Empty:
                    ping_counter += 1
                    if ping_counter % 100 == 0:
                        yield ": ping\n\n"
                    await asyncio.sleep(0.05)
                    continue

                if kind == "chunk":
                    yield f"event: chunk\ndata: {json.dumps(value)}\n\n"
                elif kind == "token":
                    yield f"event: token\ndata: {json.dumps(value)}\n\n"
                elif kind == "done":
                    final_answer = value
                    break
                elif kind == "error":
                    yield f"event: error\ndata: {json.dumps(value)}\n\n"
                    break

            await asyncio.wrap_future(future)

            if final_answer:
                done_payload = json.dumps({
                    "tokens_used": final_answer.total_tokens_used,
                    "sources": [
                        {
                            "doc_index": s.doc_index,
                            "doc_title": s.doc_title,
                            "section":   s.section,
                            "doc_type":  s.doc_type,
                            "doc_url":   s.doc_url,
                            "score":     s.score,
                        }
                        for s in final_answer.sources
                    ],
                })
                yield f"event: done\ndata: {done_payload}\n\n"

        except Exception as exc:
            logger.error("Stream error: %s", exc)
            yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Query — single-shot POST  (public)
# ---------------------------------------------------------------------------

@app.post("/query")
def query(body: QueryRequest):
    g = _get_gemini_rag(system_prompt=body.system_prompt)

    chunks = _get_rag().query(
        question=body.question,
        top_k=body.top_k,
        section_filter=body.section_filter,
        doc_type_filter=body.doc_type_filter,
    )

    result = g.answer(
        question=body.question,
        top_k=body.top_k,
        section_filter=body.section_filter,
        doc_type_filter=body.doc_type_filter,
        session_id=body.session_id,
    )

    return {
        "question":    body.question,
        "answer":      result.answer,
        "tokens_used": result.total_tokens_used,
        "chunks": [
            {
                "rank":        i + 1,
                "score":       r["score"],
                "doc_index":   r["doc_index"],
                "doc_title":   r["doc_title"],
                "section":     r["section"],
                "doc_type":    r["doc_type"],
                "doc_url":     r["doc_url"],
                "chunk_index": r["chunk_index"],
                "raw_content": r["raw_content"],
                "full_text":   r["text"],
            }
            for i, r in enumerate(chunks)
        ],
        "sources": [
            {
                "doc_index": s.doc_index,
                "doc_title": s.doc_title,
                "section":   s.section,
                "doc_type":  s.doc_type,
                "doc_url":   s.doc_url,
                "score":     s.score,
            }
            for s in result.sources
        ],
    }


# ---------------------------------------------------------------------------
# Setup routes  (no auth required)
# ---------------------------------------------------------------------------

@app.get("/admin/setup/status")
def setup_status():
    """Check if the admin password is configured."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        return {"is_set": True, "bypass": True}

    global _cached_admin_hash
    if not _cached_admin_hash:
        try:
            _cached_admin_hash = _get_admin_hash_from_db()
        except Exception as exc:
            logger.error("setup_status: Firestore read failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cannot reach database. Check Firestore IAM permissions (roles/datastore.user).",
            )

    return {"is_set": bool(_cached_admin_hash), "bypass": False}


@app.post("/admin/setup")
def setup_password(body: SetupRequest):
    """Set the initial admin password from the UI."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        return {"success": True}

    global _cached_admin_hash
    if not _cached_admin_hash:
        try:
            _cached_admin_hash = _get_admin_hash_from_db()
        except Exception as exc:
            logger.error("setup_password: Firestore read failed: %s", exc)
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail="Cannot reach database.")

    if _cached_admin_hash:
        raise HTTPException(status_code=400, detail="Password already configured.")

    hashed = password_hasher.hash(body.password)
    try:
        _set_admin_hash_in_db(hashed)
    except Exception as exc:
        logger.error("setup_password: Firestore write failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to save password. Check Firestore permissions.")
    _cached_admin_hash = hashed
    return {"success": True}


@app.post("/login")
def login(body: LoginRequest):
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        return {"access_token": "local", "token_type": "bearer"}

    global _cached_admin_hash
    if not _cached_admin_hash:
        try:
            _cached_admin_hash = _get_admin_hash_from_db()
        except Exception as exc:
            logger.error("login: Firestore read failed: %s", exc)
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Cannot reach database.")

    if not _cached_admin_hash:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Admin password not configured.")

    try:
        valid = password_hasher.verify(body.password, _cached_admin_hash)
    except Exception:
        valid = False
    if not valid:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid password.",
                            headers={"WWW-Authenticate": "Bearer"})

    secret = hashlib.sha256(_cached_admin_hash.encode()).hexdigest()
    payload = {"sub": "admin", "exp": datetime.now(timezone.utc) + timedelta(minutes=30)}
    return {"access_token": jwt.encode(payload, secret, algorithm="HS256"), "token_type": "bearer"}


@app.post("/admin/forgot-password")
async def forgot_password(request: Request):
    """Generate a password reset token and email it to admin."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "Not available in local dev mode.")

    frontend_url = os.environ.get("ADMIN_FRONTEND_URL", "").rstrip("/")
    if not frontend_url:
        logger.warning("forgot_password: ADMIN_FRONTEND_URL not set")
        return {"message": "If reset is configured, an email has been sent."}

    client_ip = request.client.host if request.client else "unknown"
    now = datetime.now(timezone.utc).timestamp()

    global _pw_reset_rate
    if client_ip in _pw_reset_rate:
        _pw_reset_rate[client_ip] = [t for t in _pw_reset_rate[client_ip]
                                      if now - t < _PW_RESET_WINDOW_SECONDS]
        if len(_pw_reset_rate[client_ip]) >= _PW_RESET_MAX_REQUESTS:
            return {"message": "If reset is configured, an email has been sent."}
        _pw_reset_rate[client_ip].append(now)
    else:
        _pw_reset_rate[client_ip] = [now]

    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    try:
        db = _firestore_client()
        db.collection("password_resets").document().set({
            "token_hash": token_hash,
            "expires_at": expires_at,
            "used": False,
            "created_at": datetime.now(timezone.utc),
            "ip": client_ip,
        })
    except Exception as exc:
        logger.error("forgot_password: Firestore write failed: %s", exc)

    reset_url = f"{frontend_url}/?token={raw_token}"
    try:
        await _send_reset_email(reset_url)
    except Exception as exc:
        logger.error("forgot_password: Email send failed: %s", exc)

    return {"message": "If reset is configured, an email has been sent."}


@app.post("/admin/reset-password")
async def reset_password(body: ResetPasswordRequest):
    """Validate reset token and update admin password."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "Not available in local dev mode.")

    token_hash = hashlib.sha256(body.token.encode()).hexdigest()

    try:
        db = _firestore_client()
        docs = list(db.collection("password_resets")
                      .where("token_hash", "==", token_hash)
                      .where("used", "==", False)
                      .where("expires_at", ">", datetime.now(timezone.utc))
                      .limit(1).stream())

        if not docs:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired reset token.")

        doc_ref = docs[0].reference

        doc_ref.update({"used": True})

        hashed = password_hasher.hash(body.new_password)
        _set_admin_hash_in_db(hashed)

        global _cached_admin_hash
        _cached_admin_hash = hashed
        # Note: _jwt_secret is no longer cached; it's recalculated on each request to avoid multi-worker issues

        return {"success": True, "message": "Password updated. Please log in with your new password."}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("reset_password: Error: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Reset failed.")


@app.get("/admin/firestore/test")
def firestore_test(_: AdminDep):
    """Diagnostic: verify Firestore read + write and report IAM issues."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        return {"ok": True, "message": "Local mode — no Firestore configured."}

    results: dict = {"database": os.environ.get("FIRESTORE_DB", "(default)")}
    try:
        db = _firestore_client()

        doc = db.collection("system_config").document("admin").get()
        results["read"] = "ok"
        results["password_set"] = doc.exists and bool(doc.to_dict().get("password_hash"))

        db.collection("system_config").document("admin").set(
            {"_diag_ping": True}, merge=True
        )
        results["write"] = "ok"

        return {"ok": True, "project": project, **results}
    except Exception as exc:
        results["error"] = str(exc)
        results["hint"] = (
            "Grant the Cloud Run service account roles/datastore.user. "
            "Run: gcloud projects add-iam-policy-binding PROJECT_ID "
            "--member=serviceAccount:SA_EMAIL --role=roles/datastore.user"
        )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=results)


@app.get("/admin/config/model")
def get_model(_: AdminDep):
    """Get the currently selected Anthropic model."""
    return {"model": _current_config.get("model", "claude-sonnet-4-6")}


@app.get("/admin/models/available")
def list_models(_: AdminDep):
    """Get list of available Anthropic chat models."""
    models = _fetch_anthropic_models()
    if not models:
        logger.warning("No models available — Anthropic API may be unreachable.")
    return {"models": models}


@app.post("/admin/config/model")
def set_model(body: SetModelRequest, _: AdminDep):
    """Update the selected Anthropic model."""
    model = body.model.strip()
    if not model:
        raise HTTPException(status_code=400, detail="Model name is required.")

    # Validate that it's a chat model (not embedding, etc.)
    available = _fetch_anthropic_models()
    available_ids = [m["id"] for m in available]
    if model not in available_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not available. Must be one of: {', '.join(available_ids)}"
        )

    # Save to Firestore and update in-memory config
    _save_model_to_firestore(model)
    _current_config["model"] = model
    logger.info("Admin updated model to: %s", model)
    return {"success": True, "model": model}


@app.get("/admin/system-prompt")
def get_system_prompt(_: AdminDep):
    """Get the current system prompt from memory (synced from GCS at startup)."""
    from rag_query import _SYSTEM_PROMPT
    return {"prompt": _SYSTEM_PROMPT, "source": "memory (synced from GCS)"}


@app.post("/admin/system-prompt")
def set_system_prompt(body: dict, _: AdminDep):
    """Update the system prompt and save to disk + GCS."""
    prompt = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    # Save to local file
    config_dir = Path("system_config")
    config_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = config_dir / "system_prompt.txt"
    try:
        prompt_file.write_text(prompt, encoding='utf-8')
        logger.info("System prompt saved to %s", prompt_file)
    except Exception as exc:
        logger.error("Failed to save system prompt: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save prompt locally.")

    # Upload to GCS if configured
    bucket = _current_config.get("gcs_bucket", "")
    if bucket:
        ok = _upload_system_prompt_to_gcs(bucket, "system_config")
        if not ok:
            logger.warning("Failed to upload system prompt to GCS, but local file saved.")

    # Update in-memory variable
    from rag_query import load_system_prompt
    load_system_prompt()
    logger.info("Admin updated system prompt")
    return {"success": True, "message": "System prompt updated and saved."}


# ---------------------------------------------------------------------------
# Admin routes  (Bearer token required)
# ---------------------------------------------------------------------------

@app.get("/admin")
def admin_redirect():
    """
    Admin UI is deployed separately on Vercel (frontend-admin/).
    For local development, it will serve the old ingest_ui.html if present.
    In production, it returns a JSON redirect message.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        html_path = Path(__file__).parent / "ingest_ui.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text(encoding="utf-8"))

    return {
        "message": "Admin UI is hosted on Vercel. Access it at your frontend-admin deployment URL.",
        "hint":    "Set the backend URL to this service's Cloud Run URL when logging in.",
    }


@app.get("/config")
def get_config(_: AdminDep):
    safe = dict(_current_config)
    if safe.get("gemini_api_key"):
        safe["gemini_api_key"] = "***set***"
    if safe.get("anthropic_api_key"):
        safe["anthropic_api_key"] = "***set***"
    return safe


@app.post("/config")
def update_config(body: ConfigUpdate, _: AdminDep):
    global _rag, _current_config
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    _current_config.update(updates)
    _rag = None
    return {"updated": list(updates.keys()), "config": get_config(_)}


@app.post("/ingest")
@_serialized_index_mutation
def ingest(body: IngestRequest, _: AdminDep):
    rag = _get_rag()
    if body.rebuild:
        chunks = rag.rebuild_index(body.url)
        action = "rebuild"
    else:
        chunks = rag.ingest_portfolio(body.url)
        action = "ingest"
    _save_and_sync(rag)
    return {"action": action, "chunks_stored": chunks, "stats": rag.stats()}


@app.get("/documents")
def list_documents(_: AdminDep):
    """Returns a list of unique documents currently indexed."""
    db = _get_rag().db
    rag = _get_rag()
    docs_by_index = {}

    # Group chunks by doc_index and get metadata
    for c in db._meta.values():
        if c.doc_index not in docs_by_index:
            docs_by_index[c.doc_index] = {
                "title":    c.doc_title,
                "doc_index": c.doc_index,
                "section":  c.section,
                "url":      c.doc_url,
                "doc_type": c.doc_type,
                "chunks":   0,
            }
        docs_by_index[c.doc_index]["chunks"] += 1

    result = list(docs_by_index.values())
    result.sort(key=lambda x: x["title"].lower())
    return result


@app.get("/documents/{doc_index}/chunks")
def get_document_chunks(doc_index: int, _: AdminDep):
    """Return full document content by concatenating all chunks in order."""
    chunks = _get_rag().query_doc(doc_index)
    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found for this doc_index.")
    # Concatenate all raw_content in chunk_index order
    content = '\n\n---\n\n'.join(c.raw_content for c in chunks)
    return {
        "doc_index": doc_index,
        "chunk_count": len(chunks),
        "content": content,
    }


@app.delete("/documents/{doc_title:path}")
@_serialized_index_mutation
def delete_document(doc_title: str, _: AdminDep):
    """Delete all chunks for a specific document title."""
    rag = _get_rag()
    count = rag.db.delete_by_doc_title(doc_title)
    _save_and_sync(rag)
    return {"deleted_chunks": count, "title": doc_title}


# ---------------------------------------------------------------------------
# Session management  (admin)
# ---------------------------------------------------------------------------

@app.get("/sessions")
def list_sessions(_: AdminDep):
    return {"sessions": _get_gemini_rag().list_sessions()}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str, _: AdminDep):
    _get_gemini_rag().clear_session(session_id)
    return {"cleared": session_id}


@app.delete("/sessions")
def clear_all_sessions(_: AdminDep):
    g = _get_gemini_rag()
    for sid in g.list_sessions():
        g.clear_session(sid)
    return {"cleared": "all"}

@app.get("/sessions/history")
def list_sessions_history(_: AdminDep, limit: int = 20, offset: int = 0):
    """Return paginated chat sessions ordered by most recent first."""
    result = _session_store.list_paginated(limit=limit, offset=offset)
    result["has_more"] = (offset + limit) < result["total"]
    return result


@app.get("/sync-jobs")
def list_sync_jobs(_: AdminDep, limit: int = Query(20, ge=1, le=100)):
    """Return recent durable auto-sync jobs."""
    return {"jobs": _list_sync_jobs(limit)}


@app.get("/sync-jobs/{job_id}")
def get_sync_job(
    job_id: str,
    _: AdminDep,
    item_limit: int = Query(100, ge=1, le=500),
    item_offset: int = Query(0, ge=0),
):
    """Return one auto-sync job, its status counts, and its items."""
    return _sync_job_status(job_id, item_limit, item_offset)


@app.post("/sync-jobs/{job_id}/pause")
def pause_sync_job(job_id: str, _: AdminDep):
    """Reset unfinished work to pending and stop new claims for a sync job."""
    return _pause_sync_job(job_id)


@app.post("/sync-jobs/{job_id}/restart")
def restart_sync_job(job_id: str, _: AdminDep):
    """Resume a paused job after pause has reset its unfinished items."""
    return _restart_sync_job(job_id)


@app.post("/sync-jobs/{job_id}/remove-videos")
def remove_sync_job_videos(job_id: str, _: AdminDep):
    """Remove non-processing video items from a Google Drive sync job."""
    return _remove_gdrive_video_items(job_id)


@app.delete("/sync-jobs/{job_id}/items/{item_id}")
def remove_sync_job_item(job_id: str, item_id: str, _: AdminDep):
    """Remove a pending or failed item from a sync job without indexing it."""
    return _remove_sync_job_item(job_id, item_id)


# ---------------------------------------------------------------------------
# Additional ingest endpoints  (admin)
# ---------------------------------------------------------------------------

@app.post("/ingest/folder")
@_serialized_index_mutation
def ingest_folder(
    _: AdminDep,
    section: str = Form("general"),
    recursive: bool = Form(True),
    file: UploadFile = File(...),
):
    """
    Ingest files from an uploaded zip, or a single supported document.

    Supported single files: .pdf .docx .doc .odt .pptx .ppt
                            .xlsx .xls .xlsm .xlsb .csv .ods
                            .txt .md .markdown .rst .html .htm
    Zip files: any zip containing any mix of the above.
    """
    from orchestrator import _ALL_SUPPORTED
    rag = _get_rag()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_bytes = file.file.read()
            extract_dir = os.path.join(tmpdir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)

            filename = getattr(file, "filename", "uploaded_file")
            suffix = Path(filename).suffix.lower()

            if suffix in _ALL_SUPPORTED:
                # Single supported file — write directly into the extract dir
                dest = os.path.join(extract_dir, filename)
                with open(dest, "wb") as f:
                    f.write(file_bytes)
            elif suffix == ".zip" or filename.lower().endswith(".zip"):
                zip_path = os.path.join(tmpdir, "uploaded.zip")
                with open(zip_path, "wb") as f:
                    f.write(file_bytes)
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                except zipfile.BadZipFile:
                    raise HTTPException(status_code=400, detail="Invalid zip file.")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type '{suffix}'. Upload a .zip or a supported document."
                )

            chunks = rag.ingest_folder(
                folder_path=extract_dir,
                section=section,
                recursive=recursive,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _save_and_sync(rag)
    return {
        "action":        "ingest_folder",
        "folder_path":   file.filename,
        "section":       section,
        "chunks_stored": chunks,
        "stats":         rag.stats(),
    }


@app.post("/ingest/documents")
@_serialized_index_mutation
def ingest_documents(body: RawDocumentsRequest, _: AdminDep):
    """
    Inject pre-written text documents directly into the index.

    Each document goes through the normal chunker pipeline.
    Useful for adding custom bios, CVs, notes, or any text
    that doesn't have a URL to scrape.
    """
    rag = _get_rag()
    docs = [d.model_dump() for d in body.documents]
    chunks = rag.ingest_raw_documents(docs)
    _save_and_sync(rag)
    return {
        "action":        "ingest_documents",
        "docs_received": len(docs),
        "chunks_stored": chunks,
        "stats":         rag.stats(),
    }


@app.post("/ingest/videos")
@_serialized_index_mutation
def ingest_videos(body: VideosIngestRequest, _: AdminDep):
    """
    Summarise a list of YouTube video URLs or playlist URLs via Gemini.

    Playlist URLs (youtube.com/playlist?list=…) are automatically expanded
    to individual videos.  Results are cached so replaying is free.
    """
    rag = _get_rag()
    chunks = rag.ingest_videos(body.urls, section=body.section)
    _save_and_sync(rag)
    return {
        "action":        "ingest_videos",
        "urls_received": len(body.urls),
        "chunks_stored": chunks,
        "stats":         rag.stats(),
    }


# ---------------------------------------------------------------------------
# Cleanup  (admin)
# ---------------------------------------------------------------------------

@app.post("/cleanup/preview")
def cleanup_preview(body: CleanupRequest, _: AdminDep):
    """
    Dry-run: return how many chunks would be deleted and up to 20 samples.
    No changes are made to the index.
    """
    db = _get_rag().db
    flagged, samples = _scan_cleanup_candidates(db, body)
    return {
        "would_delete": len(flagged),
        "total_chunks": len(db._meta),
        "samples":      samples,
    }


@app.post("/cleanup/apply")
@_serialized_index_mutation
def cleanup_apply(body: CleanupRequest, _: AdminDep):
    """
    Apply the same filters as /cleanup/preview and permanently delete matched chunks.
    Saves the updated index to disk (and GCS if configured).
    """
    rag = _get_rag()
    flagged, _ = _scan_cleanup_candidates(rag.db, body)
    if flagged:
        rag.db._remove_int_ids(flagged)
        _save_and_sync(rag)
    return {
        "deleted_chunks": len(flagged),
        "deleted_cache":  0,
        "total_chunks":   len(rag.db._meta),
    }


# ---------------------------------------------------------------------------
# Danger zone  (admin)
# ---------------------------------------------------------------------------

@app.delete("/index")
@_serialized_index_mutation
def clear_index(_: AdminDep):
    rag = _get_rag()
    rag.db.clear()
    _save_and_sync(rag)
    return {"cleared": "faiss_index"}


# ---------------------------------------------------------------------------
# YouTube PubSubHubbub  (automatic new-video ingestion)
# ---------------------------------------------------------------------------

@app.get("/youtube/notify")
async def youtube_verify(
    hub_challenge: str = Query(..., alias="hub.challenge"),
    hub_topic: Optional[str] = Query(None, alias="hub.topic"),
):
    """YouTube calls this once on subscription to verify the endpoint is real."""
    if hub_topic:
        channel_id = hub_topic.rsplit("channel_id=", 1)[-1]
        if channel_id and channel_id != hub_topic:
            try:
                _sync_jobs_client().collection(_YOUTUBE_SUBSCRIPTIONS_COLLECTION).document(channel_id).set({
                    "channel_id": channel_id,
                    "topic": hub_topic,
                    "status": "verified",
                    "verified_at": _sync_timestamp(),
                    "updated_at": _sync_timestamp(),
                }, merge=True)
            except Exception:
                logger.exception("Unable to record YouTube subscription verification for %s", channel_id)
    return PlainTextResponse(hub_challenge)


@app.post("/youtube/notify")
async def youtube_notify(request: Request):
    """
    YouTube POSTs an Atom XML payload here within seconds of a new upload.
    It records video IDs durably and returns before background ingestion begins.
    """
    body = await request.body()

    secret = os.environ.get("PUBSUB_SECRET", "")
    if secret:
        sig = request.headers.get("X-Hub-Signature", "")
        expected = "sha1=" + hmac.new(secret.encode(), body, hashlib.sha1).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YouTube notification XML: {exc}")

    ns = {"yt": "http://www.youtube.com/xml/schemas/2015"}
    atom_namespace = "{http://www.w3.org/2005/Atom}"
    items = []

    for entry in root.iter(f"{atom_namespace}entry"):
        vid_el = entry.find("yt:videoId", ns)
        if vid_el is not None and vid_el.text:
            video_id = vid_el.text.strip()
            if not video_id:
                continue
            title = entry.findtext(f"{atom_namespace}title") or video_id
            published_at = (
                entry.findtext(f"{atom_namespace}published")
                or entry.findtext(f"{atom_namespace}updated")
            )
            items.append({
                "source_id": video_id,
                "name": title.strip(),
                "created_time": published_at,
                "metadata": {"title": title.strip()},
            })

    if not items:
        return Response(status_code=204)

    try:
        job_id = _create_sync_job("youtube", metadata={"notification_received_at": _sync_timestamp()})
        queued = _add_sync_job_items(job_id, "youtube", items)
    except Exception:
        logger.exception("Unable to queue YouTube notification items")
        raise HTTPException(status_code=503, detail="Unable to queue YouTube notification.")

    logger.info("PubSubHubbub: notification queued job=%s discovered=%d queued=%d", job_id, len(items), queued)
    _wake_sync_worker()

    return Response(status_code=204)


@app.post("/youtube/resubscribe")
def youtube_resubscribe(request: Request):
    """
    Re-registers all watched channels with PubSubHubbub.
    Hit once manually to bootstrap, then Cloud Scheduler calls it every 15 days.
    """
    

    channel_ids = [
        c.strip()
        for c in os.environ.get("WATCHED_CHANNEL_IDS", "").split(",")
        if c.strip()
    ]
    if not channel_ids:
        return {"resubscribed": [], "warning": "WATCHED_CHANNEL_IDS not set"}

    callback = str(request.base_url).rstrip("/") + "/youtube/notify"
    secret   = os.environ.get("PUBSUB_SECRET", "")

    results = []
    db_fs = None
    try:
        db_fs = _sync_jobs_client()
    except Exception:
        logger.exception("Unable to initialize Firestore for YouTube subscription tracking")

    for cid in channel_ids:
        topic = f"https://www.youtube.com/xml/feeds/videos.xml?channel_id={cid}"
        data  = {
            "hub.mode":          "subscribe",
            "hub.topic":         topic,
            "hub.callback":      callback,
            "hub.lease_seconds": 2592000,   # 30 days (YouTube's max)
        }
        if secret:
            data["hub.secret"] = secret
        try:
            resp = _req.post("https://pubsubhubbub.appspot.com/subscribe", data=data, timeout=15)
            request_status = "requested" if resp.status_code == 202 else "rejected"
            result = {"channel_id": cid, "http_status": resp.status_code, "status": request_status}
        except Exception as exc:
            logger.error("PubSubHubbub subscribe request failed for %s: %s", cid, exc)
            result = {"channel_id": cid, "status": "request_failed", "error": str(exc)}

        results.append(result)
        logger.info("PubSubHubbub subscribe: channel=%s status=%s", cid, result["status"])
        if db_fs is not None:
            try:
                db_fs.collection(_YOUTUBE_SUBSCRIPTIONS_COLLECTION).document(cid).set({
                    "channel_id": cid,
                    "topic": topic,
                    "callback": callback,
                    "status": result["status"],
                    "http_status": result.get("http_status"),
                    "error": result.get("error"),
                    "requested_at": _sync_timestamp(),
                    "updated_at": _sync_timestamp(),
                }, merge=True)
            except Exception:
                logger.exception("Unable to record YouTube subscription request for %s", cid)

    return {"resubscribed": results}


# ---------------------------------------------------------------------------
# Google Drive weekly sync (automatic new-file ingestion)
# ---------------------------------------------------------------------------
@app.post("/gdrive/sync")
def gdrive_sync():
    """
    Discover newly created supported files and queue them for background ingestion.
    The last_sync_time checkpoint is advanced only after all items are durable.
    Called weekly by Cloud Scheduler.
    """
    if not GDRIVE_FOLDER_ID:
        return {"synced": 0, "warning": "GDRIVE_FOLDER_ID not set"}

    try:
        drive_service = _get_gdrive_service()
    except Exception as e:
        logger.error(f"gdrive_sync: Failed to initialize Drive service: {e}")
        raise HTTPException(status_code=503, detail=f"Google Drive service unavailable: {e}")

    try:
        folder = drive_service.files().get(
            fileId=GDRIVE_FOLDER_ID,
            fields="id,name,mimeType",
        ).execute()
        folder_info = {"id": folder["id"], "name": folder.get("name")}
        logger.info("gdrive_sync: folder_id=%s folder_name=%s",
                    folder_info["id"], folder_info["name"])
    except Exception as e:
        logger.error("gdrive_sync: Failed to read Drive folder %s: %s", GDRIVE_FOLDER_ID, e)
        raise HTTPException(status_code=502, detail=f"Google Drive folder unavailable: {e}")

    last_sync_time = None
    project = os.environ.get('GOOGLE_CLOUD_PROJECT', '')
    if project:
        sync_state = _get_gdrive_sync_state()
        if sync_state:
            last_sync_time = sync_state.get('last_sync_time')

    logger.info("gdrive_sync: last_sync_time=%s", last_sync_time or "none")

    query = f"'{GDRIVE_FOLDER_ID}' in parents and trashed = false"
    if last_sync_time:
        query += f" and createdTime > '{last_sync_time}'"

    all_files = []
    page_token = None
    try:
        while True:
            results = drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType, size, createdTime)',
                pageSize=100,
                pageToken=page_token,
                orderBy='createdTime desc'
            ).execute()
            all_files.extend(results.get('files', []))
            page_token = results.get('nextPageToken')
            if not page_token:
                break
    except Exception as e:
        logger.error(f"gdrive_sync: Google Drive API error: {e}")
        raise HTTPException(status_code=502, detail=f"Google Drive API error: {e}")

    from orchestrator import _ALL_SUPPORTED
    new_files = [
        item for item in all_files
        if (
            Path(item.get('name', '')).suffix.lower() in _ALL_SUPPORTED
            and item.get('mimeType', '').startswith('audio/')
        )
    ]

    if not new_files:
        return {
            "synced": 0,
            "message": "No new supported files since last sync",
            "last_sync_time": last_sync_time,
            "folder": folder_info,
            "files_matched": len(all_files),
        }

    newest_created_time = max(item["createdTime"] for item in new_files)
    job_id = _create_sync_job(
        "gdrive",
        discovered_after=last_sync_time,
        metadata={"folder": folder_info, "files_matched": len(all_files)},
    )
    queued = _add_sync_job_items(job_id, "gdrive", [
        {
            "source_id": item["id"],
            "name": item["name"],
            "created_time": item["createdTime"],
            "metadata": {
                "mime_type": item.get("mimeType"),
                "size": item.get("size"),
                "folder_id": GDRIVE_FOLDER_ID,
            },
        }
        for item in new_files
    ])

    if project:
        _save_gdrive_sync_state(newest_created_time)

    _wake_sync_worker()
    return {
        "job_id": job_id,
        "queued": queued,
        "discovered": len(new_files),
        "last_sync_time": last_sync_time,
        "next_sync_after": newest_created_time,
        "folder": folder_info,
    }

# ---------------------------------------------------------------------------
# Blog weekly sync  (automatic new-blog ingestion)
# ---------------------------------------------------------------------------
@app.post("/blogs/sync")
def blogs_sync():
    """
    Discover new or updated CMS blogs and enqueue their ingestion.

    The background worker processes one queued blog at a time, so the scheduler
    request completes without waiting for PDF download, extraction, or indexing.
    """

    import urllib3

    # Suppress insecure request warnings caused by the expired CMS SSL cert
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    last_sync_time: Optional[str] = None
    db_fs = _sync_jobs_client()
    try:
        doc = db_fs.collection("system_config").document("blog_sync").get()
        if doc.exists:
            last_sync_time = doc.to_dict().get("last_sync_time")
    except Exception as exc:
        logger.error("blogs_sync: Firestore checkpoint read failed: %s", exc)
        raise HTTPException(status_code=503, detail="Blog sync checkpoint is unavailable.")

    try:
        last_sync_datetime = _parse_sync_datetime(last_sync_time) if last_sync_time else None
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid blog sync checkpoint: {exc}")

    api_url = os.environ.get("SWATI_DESAI_API")
    if not api_url:
        raise HTTPException(status_code=503, detail="SWATI_DESAI_API is not configured.")
    headers = {"accept": "text/plain", "Content-Type": "application/json"}

    all_blogs = []
    seen_blog_versions = set()
    page_index = 0
    page_size = 100
    while True:
        try:
            resp = _req.post(
                api_url,
                json={"pageIndex": page_index, "pageSize": page_size},
                headers=headers,
                verify=False,
                timeout=30,
            )
            resp.raise_for_status()
            page_blogs = resp.json().get("data", [])
        except Exception as exc:
            logger.error("blogs_sync: CMS API connectivity error: %s", exc)
            raise HTTPException(status_code=502, detail=f"Blog CMS API error: {exc}")

        if not isinstance(page_blogs, list):
            raise HTTPException(status_code=502, detail="Blog CMS API returned an invalid data payload.")

        unique_on_page = 0
        for blog in page_blogs:
            blog_id = str(blog.get("id", "")).strip()
            blog_version = blog.get("updatedDate") or blog.get("createdDate") or ""
            dedupe_key = f"{blog_id}:{blog_version}"
            if not blog_id or dedupe_key in seen_blog_versions:
                continue
            seen_blog_versions.add(dedupe_key)
            all_blogs.append(blog)
            unique_on_page += 1

        if len(page_blogs) < page_size or unique_on_page == 0:
            break
        page_index += 1

    new_blogs = []

    newest_timestamp = last_sync_datetime
    for blog in all_blogs:
        blog_id = str(blog.get("id", "")).strip()
        updated_date_str = blog.get("updatedDate") or blog.get("createdDate")
        if not blog_id or not updated_date_str:
            continue
        try:
            blog_datetime = _parse_sync_datetime(updated_date_str)
        except ValueError:
            logger.warning("blogs_sync: skipping %s with invalid timestamp %r", blog_id, updated_date_str)
            continue
        if last_sync_datetime is None or blog_datetime > last_sync_datetime:
            new_blogs.append((blog, blog_datetime))
            if newest_timestamp is None or blog_datetime > newest_timestamp:
                newest_timestamp = blog_datetime

    if not new_blogs:
        return {
            "queued": 0,
            "message": "No new or updated blog entries discovered since last sync.",
            "last_sync_time": last_sync_time,
        }

    job_id = _create_sync_job(
        "blogs",
        discovered_after=last_sync_time,
        metadata={"cms_api": api_url},
    )
    items = []
    for blog, blog_datetime in new_blogs:
        blog_id = str(blog["id"])
        title = blog.get("name") or "Untitled Blog"
        content = "\n\n".join(filter(None, [
            (blog.get("subDescription") or "").strip(),
            (blog.get("description") or "").strip(),
        ])).strip()
        items.append({
            "source_id": blog_id,
            "dedupe_key": f"{blog_id}:{blog_datetime.isoformat()}",
            "name": title,
            "created_time": blog_datetime.isoformat(),
            "metadata": {
                "title": title,
                "content": content,
                "pdf_url": (blog.get("document") or "").strip(),
                "updated_time": blog_datetime.isoformat(),
            },
        })

    try:
        queued = _add_sync_job_items(job_id, "blogs", items)
        db_fs.collection("system_config").document("blog_sync").set({
            "last_sync_time": newest_timestamp.isoformat(),
            "execution_ran_at": _sync_timestamp(),
        }, merge=True)
    except Exception:
        logger.exception("blogs_sync: unable to persist discovered blog items")
        raise HTTPException(status_code=503, detail="Unable to queue blog sync items.")

    _wake_sync_worker()
    return {
        "job_id": job_id,
        "queued": queued,
        "discovered": len(new_blogs),
        "last_sync_time": last_sync_time,
        "next_sync_after": newest_timestamp.isoformat(),
    }
