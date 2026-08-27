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
  gunicorn -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120

Session persistence
-------------------
If GOOGLE_CLOUD_PROJECT is set, chat histories are stored in Firestore
(collection: rag_sessions).  Otherwise an in-memory dict is used — fine
for local development, but histories are lost on restart.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import hmac
import json
import logging
import os
import secrets
import queue as _sync_queue
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
        found   = 0
        for fname in _GCS_INDEX_FILES:
            blob = bucket.blob(f"rag_index/{fname}")
            if blob.exists():
                blob.download_to_filename(str(path / fname))
                logger.info("GCS ↓ downloaded %s", fname)
                found += 1
            else:
                logger.warning("GCS: %s not found in bucket %s", fname, bucket_name)
        return found == len(_GCS_INDEX_FILES)
    except Exception as exc:
        logger.error("GCS download failed: %s", exc)
        return False


def _upload_index_to_gcs(bucket_name: str, index_dir: str) -> None:
    """
    Upload faiss.index + metadata.pkl from index_dir to GCS.
    Called after every rag.save() so the index survives container restarts.
    """
    try:
        client = _gcs_client()
        bucket = client.bucket(bucket_name)
        path   = Path(index_dir)
        for fname in _GCS_INDEX_FILES:
            local = path / fname
            if local.exists():
                bucket.blob(f"rag_index/{fname}").upload_from_filename(str(local))
                logger.info("GCS ↑ uploaded %s", fname)
            else:
                logger.warning("GCS upload: %s not found locally", fname)
    except Exception as exc:
        logger.error("GCS upload failed: %s", exc)


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
    global _cached_admin_hash, _jwt_secret

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

    if not _jwt_secret:
        _jwt_secret = hashlib.sha256(_cached_admin_hash.encode()).hexdigest()

    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        jwt.decode(creds.credentials, _jwt_secret, algorithms=["HS256"])
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
    "gcs_bucket":        os.environ.get("GCS_BUCKET", ""),  # e.g. "bhagwan-rag-store"
}

_current_config: dict             = dict(_DEFAULT_CONFIG)
_rag:            Optional[RAGOrchestrator] = None

# Model cache: {models: [...], timestamp: ...}
_model_cache: dict = {"models": None, "timestamp": None}
_MODEL_CACHE_TTL = 300  # 5 minutes

# Microsoft Graph token cache: {token: ..., timestamp: ...}
_graph_token_cache: dict = {"token": None, "timestamp": None}
_GRAPH_TOKEN_CACHE_TTL = 3600  # 1 hour (tokens typically valid for 1 hour)

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


def _get_microsoft_graph_token() -> str:
    """
    Get OAuth2 access token for Microsoft Graph API using client credentials flow.
    Caches token for 1 hour. Raises HTTPException on failure.
    """
    now = datetime.now(timezone.utc)

    # Return cached token if still valid
    if (_graph_token_cache["token"] is not None and
        _graph_token_cache["timestamp"] is not None and
        (now - _graph_token_cache["timestamp"]).total_seconds() < _GRAPH_TOKEN_CACHE_TTL):
        return _graph_token_cache["token"]

    client_id = os.environ.get("MICROSOFT_CLIENT_ID", "").strip()
    client_secret = os.environ.get("MICROSOFT_CLIENT_SECRET", "").strip()
    tenant_id = os.environ.get("MICROSOFT_TENANT_ID", "").strip()

    if not (client_id and client_secret and tenant_id):
        logger.error("Missing Microsoft OAuth2 credentials: CLIENT_ID=%s, SECRET=%s, TENANT=%s",
                     bool(client_id), bool(client_secret), bool(tenant_id))
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                          detail="OneDrive authentication not configured")

    try:
        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://graph.microsoft.com/.default",
        }
        response = _req.post(token_url, data=payload, timeout=10)
        response.raise_for_status()

        token_data = response.json()
        token = token_data.get("access_token")
        if not token:
            raise ValueError("No access_token in response")

        # Cache the token
        _graph_token_cache["token"] = token
        _graph_token_cache["timestamp"] = now
        logger.info("Obtained new Microsoft Graph access token")

        return token
    except Exception as exc:
        logger.error("Failed to obtain Microsoft Graph token: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                          detail=f"OneDrive authentication failed: {exc}")


def _get_gdrive_service():
    """Initialize authenticated Google Drive service with caching."""
    global _gdrive_service_cache, _gdrive_service_cache_time
    now = datetime.now(timezone.utc)

    if (_gdrive_service_cache is not None and
        _gdrive_service_cache_time is not None and
        (now - _gdrive_service_cache_time).total_seconds() < _GDRIVE_SERVICE_CACHE_TTL):
        return _gdrive_service_cache

    try:
        credentials = service_account.Credentials.from_service_account_file(
            GDRIVE_SERVICE_ACCOUNT_FILE,
            scopes=GDRIVE_SCOPES
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

    try:
        db_fs = _firestore_client()
        db_fs.collection('system_config').document('gdrive_sync').set(
            {'last_sync_time': last_sync_time},
            merge=True
        )
        logger.info(f"Saved Google Drive sync state: {last_sync_time}")
    except Exception as e:
        logger.warning(f"Failed to save Google Drive sync state: {e}")


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

    # --- 3. Warm up RAG (load FAISS index + rebuild BM25 before first request) ---
    try:
        _get_rag()
        logger.info("Startup: RAG index loaded and ready.")
    except Exception as exc:
        logger.error("Startup: RAG warm-up failed — first query will trigger lazy load. Error: %s", exc)


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

        global _cached_admin_hash, _jwt_secret
        _cached_admin_hash = hashed
        _jwt_secret = hashlib.sha256(hashed.encode()).hexdigest()

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
    docs = {}
    for c in db._meta.values():
        title = c.doc_title
        if title not in docs:
            docs[title] = {
                "title":    title,
                "section":  c.section,
                "url":      c.doc_url,
                "doc_type": c.doc_type,
                "chunks":   0,
            }
        docs[title]["chunks"] += 1
    result = list(docs.values())
    result.sort(key=lambda x: x["title"].lower())
    return result


@app.delete("/documents/{doc_title:path}")
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


# ---------------------------------------------------------------------------
# Additional ingest endpoints  (admin)
# ---------------------------------------------------------------------------

@app.post("/ingest/folder")
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
def clear_index(_: AdminDep):
    _get_rag().db.clear()
    return {"cleared": "faiss_index"}


# ---------------------------------------------------------------------------
# YouTube PubSubHubbub  (automatic new-video ingestion)
# ---------------------------------------------------------------------------

@app.get("/youtube/notify")
async def youtube_verify(hub_challenge: str = Query(..., alias="hub.challenge")):
    """YouTube calls this once on subscription to verify the endpoint is real."""
    return PlainTextResponse(hub_challenge)


@app.post("/youtube/notify")
async def youtube_notify(request: Request):
    """
    YouTube POSTs an Atom XML payload here within seconds of a new upload.
    Parses the video ID and runs the existing ingest_videos pipeline.
    """
    body = await request.body()

    secret = os.environ.get("PUBSUB_SECRET", "")
    if secret:
        sig = request.headers.get("X-Hub-Signature", "")
        expected = "sha1=" + hmac.new(secret.encode(), body, hashlib.sha1).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise HTTPException(status_code=403, detail="Invalid signature")

    root = ET.fromstring(body)
    ns = {"yt": "http://www.youtube.com/xml/schemas/2015"}
    for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
        vid_el = entry.find("yt:videoId", ns)
        if vid_el is not None and vid_el.text:
            url = f"https://www.youtube.com/watch?v={vid_el.text.strip()}"
            logger.info("PubSubHubbub: new video detected %s", url)
            rag = _get_rag()
            rag.ingest_videos([url])
            _save_and_sync(rag)

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
        resp = _req.post("https://pubsubhubbub.appspot.com/subscribe", data=data, timeout=15)
        results.append({"channel_id": cid, "http_status": resp.status_code})
        logger.info("PubSubHubbub subscribe: channel=%s status=%d", cid, resp.status_code)

    return {"resubscribed": results}


# ---------------------------------------------------------------------------
# OneDrive weekly sync  (automatic new-file ingestion)
# ---------------------------------------------------------------------------
@app.post("/onedrive/sync")
def onedrive_sync():
    """
    List files in the shared OneDrive folder, ingest any added/modified .m4a files
    since the last sync, and update the last_sync_time in Firestore.
    Called weekly by Cloud Scheduler.
    """
    import base64

    share_url = os.environ.get("ONEDRIVE_SHARE_URL", "")
    if not share_url:
        return {"synced": 0, "warning": "ONEDRIVE_SHARE_URL not set"}

    # Get OAuth2 token for Graph API authentication
    token = _get_microsoft_graph_token()
    headers = {"Authorization": f"Bearer {token}"}

    encoded = base64.urlsafe_b64encode(("u!" + share_url).encode()).decode().rstrip("=")
    try:
        resp = _req.get(
            f"https://graph.microsoft.com/v1.0/shares/{encoded}/driveItem/children",
            params={"$select": "id,name,lastModifiedDateTime,file,@microsoft.graph.downloadUrl"},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.error("onedrive_sync: Graph API error: %s", exc)
        raise HTTPException(status_code=502, detail=f"OneDrive API error: {exc}")

    # MODIFICATION: Strictly filter for files ending in .m4a (case-insensitive)
    all_files = [
        item for item in resp.json().get("value", []) 
        if "file" in item and item.get("name", "").lower().endswith(".m4a")
    ]

    last_sync_time: Optional[str] = None
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if project:
        try:
            db_fs = _firestore_client()
            doc = db_fs.collection("system_config").document("onedrive_sync").get()
            if doc.exists:
                last_sync_time = doc.to_dict().get("last_sync_time")
        except Exception as exc:
            logger.warning("onedrive_sync: Firestore read failed: %s", exc)

    new_files = [
        f for f in all_files
        if last_sync_time is None or f["lastModifiedDateTime"] > last_sync_time
    ]

    if not new_files:
        return {"synced": 0, "message": "No new .m4a files since last sync"}

    rag = _get_rag()
    total_chunks = 0
    synced_names = []
    max_ingested_time = last_sync_time

    with tempfile.TemporaryDirectory() as tmpdir:
        for item in new_files:
            download_url = item.get("@microsoft.graph.downloadUrl")
            if not download_url:
                logger.warning("onedrive_sync: no download URL for %s", item["name"])
                continue
            try:
                # Stream the download just to be safe with memory
                response = _req.get(download_url, headers=headers, stream=True, timeout=60)
                response.raise_for_status()

                dest = os.path.join(tmpdir, item["name"])
                with open(dest, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                synced_names.append(item["name"])
                max_ingested_time = item["lastModifiedDateTime"]
                logger.info("onedrive_sync: downloaded %s", item["name"])
            except Exception as exc:
                logger.warning("onedrive_sync: failed to download %s: %s", item["name"], exc)

        if synced_names:
            # rag.ingest_folder will now only see the .m4a files downloaded
            total_chunks = rag.ingest_folder(tmpdir, section="onedrive", recursive=False)

    if total_chunks > 0:
        _save_and_sync(rag)

    if project and synced_names:
        try:
            db_fs = _firestore_client()
            db_fs.collection("system_config").document("onedrive_sync").set(
                {"last_sync_time": max_ingested_time}, merge=True
            )
        except Exception as exc:
            logger.warning("onedrive_sync: Firestore write failed: %s", exc)

    return {"synced": len(synced_names), "chunks_stored": total_chunks, "files": synced_names}

# ---------------------------------------------------------------------------
# Google Drive weekly sync (automatic new-file ingestion)
# ---------------------------------------------------------------------------
@app.post("/gdrive/sync")
def gdrive_sync():
    """
    List files in the shared Google Drive folder, ingest any added/modified files
    since the last sync, and update the last_sync_time in Firestore.
    Called weekly by Cloud Scheduler.
    """
    if not GDRIVE_FOLDER_ID:
        return {"synced": 0, "warning": "GDRIVE_FOLDER_ID not set"}

    try:
        drive_service = _get_gdrive_service()
    except Exception as e:
        logger.error(f"gdrive_sync: Failed to initialize Drive service: {e}")
        raise HTTPException(status_code=503, detail=f"Google Drive service unavailable: {e}")

    # Get last sync time
    last_sync_time = None
    project = os.environ.get('GOOGLE_CLOUD_PROJECT', '')

    if project:
        sync_state = _get_gdrive_sync_state()
        if sync_state:
            last_sync_time = sync_state.get('last_sync_time')

    # Build query for files
    query = f"'{GDRIVE_FOLDER_ID}' in parents and trashed = false"
    if last_sync_time:
        query += f" and modifiedTime > '{last_sync_time}'"

    # List files (with pagination for folders with >100 files)
    all_files = []
    page_token = None
    try:
        while True:
            results = drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType, size, modifiedTime)',
                pageSize=100,
                pageToken=page_token,
                orderBy='modifiedTime desc'
            ).execute()
            all_files.extend(results.get('files', []))
            page_token = results.get('nextPageToken')
            if not page_token:
                break
    except Exception as e:
        logger.error(f"gdrive_sync: Google Drive API error: {e}")
        raise HTTPException(status_code=502, detail=f"Google Drive API error: {e}")

    # Filter for supported file types
    supported_types = [
        'application/pdf',
        'text/plain',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'audio/mpeg',
        'audio/mp4',
        'application/x-m4a'
    ]

    new_files = [f for f in all_files if f.get('mimeType') in supported_types]

    if not new_files:
        return {"synced": 0, "message": "No new supported files since last sync"}

    rag = _get_rag()
    total_chunks = 0
    synced_names = []
    # Initialize max_ingested_time: use current time (no microseconds) if first sync, otherwise use last_sync_time
    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat() + 'Z'
    max_ingested_time = last_sync_time if last_sync_time else now_utc

    with tempfile.TemporaryDirectory() as tmpdir:
        for item in new_files:
            file_id = item['id']
            file_name = item['name']

            try:
                # Download file
                file_bytes = _download_gdrive_file(drive_service, file_id)

                # Save to temp directory
                dest = os.path.join(tmpdir, file_name)
                with open(dest, 'wb') as f:
                    f.write(file_bytes)

                synced_names.append(file_name)
                max_ingested_time = item['modifiedTime']
                logger.info(f"gdrive_sync: downloaded {file_name}")

            except Exception as e:
                logger.warning(f"gdrive_sync: failed to download {file_name}: {e}")
                continue

        # Ingest files
        if synced_names:
            try:
                total_chunks = rag.ingest_folder(tmpdir, section='gdrive', recursive=False)
            except Exception as e:
                logger.error(f"gdrive_sync: Ingestion failed: {e}")
                return {
                    "synced": len(synced_names),
                    "chunks_stored": 0,
                    "files": synced_names,
                    "error": f"Ingestion failed: {str(e)}"
                }

    # Save index and update sync state ONLY if ingestion succeeded
    if total_chunks > 0:
        try:
            _save_and_sync(rag)
        except Exception as e:
            logger.error(f"gdrive_sync: Failed to save index to GCS: {e}")
            return {
                "synced": len(synced_names),
                "chunks_stored": total_chunks,
                "files": synced_names,
                "error": f"GCS save failed: {str(e)}"
            }
        # Only save sync state AFTER successful GCS upload
        if project:
            _save_gdrive_sync_state(max_ingested_time)

    return {
        "synced": len(synced_names),
        "chunks_stored": total_chunks,
        "files": synced_names,
        "next_sync_after": max_ingested_time
    }

# ---------------------------------------------------------------------------
# Blog weekly sync  (automatic new-blog ingestion)
# ---------------------------------------------------------------------------
@app.post("/blogs/sync")
def blogs_sync():
    """
    Fetch blogs from the external CMS, filter for new posts since the last run,
    and process them natively using the appropriate RAG pipeline strategy.
    Called weekly by Cloud Scheduler.
    """

    import urllib3

    # Suppress insecure request warnings caused by the expired CMS SSL cert
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # 1. Look up the last execution checkpoint from Firestore
    last_sync_time: Optional[str] = None
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    
    if project:
        try:
            db_fs = _firestore_client()
            doc = db_fs.collection("system_config").document("blog_sync").get()
            if doc.exists:
                last_sync_time = doc.to_dict().get("last_sync_time")
        except Exception as exc:
            logger.warning("blogs_sync: Firestore read failed: %s", exc)

    # Convert checkpoint string to datetime if it exists
    last_sync_datetime = None
    if last_sync_time:
        last_sync_datetime = datetime.fromisoformat(last_sync_time.split("+")[0])

    # 2. Query the external CMS API
    api_url = os.environ.get("SWATI_DESAI_API")
    payload = {"pageIndex": 0, "pageSize": 100}
    headers = {"accept": "text/plain", "Content-Type": "application/json"}

    try:
        resp = _req.post(api_url, json=payload, headers=headers, verify=False, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("blogs_sync: CMS API connectivity error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Blog CMS API error: {exc}")

    all_blogs = resp.json().get("data", [])
    new_blogs = []

    # Filter out entries already processed
    for blog in all_blogs:
        updated_date_str = blog.get("updatedDate") or blog.get("createdDate")
        if not updated_date_str:
            continue
        
        blog_datetime = datetime.fromisoformat(updated_date_str)
        if last_sync_datetime is None or blog_datetime > last_sync_datetime:
            new_blogs.append((blog, blog_datetime))

    if not new_blogs:
        return {"synced": 0, "message": "No new blog entries discovered since last sync."}

    rag = _get_rag()
    total_chunks = 0
    processed_titles = []
    newest_timestamp = last_sync_datetime or datetime.fromisoformat("1970-01-01T00:00:00")

    # Arrays to isolate batch workloads
    raw_text_docs = []

    # 3. Process each unique new blog
    for blog, blog_datetime in new_blogs:
        title = blog.get("name", "Untitled Blog")
        pdf_url = blog.get("document")

        # --- Pipeline Strategy Selection ---
        if pdf_url and pdf_url.strip():
            # STRATEGY A: PDF Document Present -> Use Folder Ingestion Pipeline Logic
            logger.info("blogs_sync: processing via PDF pipeline -> %s", title)
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    pdf_resp = _req.get(pdf_url, stream=True, timeout=60)
                    pdf_resp.raise_for_status()
                    
                    # Create a clean safe filename from the blog ID or Title
                    safe_filename = f"{blog.get('id', 'doc')}.pdf"
                    dest = os.path.join(tmpdir, safe_filename)
                    
                    with open(dest, "wb") as f:
                        for chunk in pdf_resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Natively route through folder chunker pipeline
                    chunks = rag.ingest_folder(tmpdir, section="blogs", recursive=False)
                    total_chunks += chunks
                    processed_titles.append(f"[PDF] {title}")
            except Exception as exc:
                logger.error("blogs_sync: failed to download document for %s: %s", title, exc)
                continue
        else:
            # STRATEGY B: No Document -> Compile Text Data and Use Raw Document Pipeline
            logger.info("blogs_sync: processing via Raw Text pipeline -> %s", title)
            sub_desc = blog.get("subDescription") or ""
            desc = blog.get("description") or ""
            
            # Combine the body text components natively
            full_text = f"{sub_desc}\n\n{desc}".strip()
            
            raw_text_docs.append({
                "title": title,
                "text": full_text if full_text else "No content available."
            })
            processed_titles.append(f"[Text] {title}")

        # Track the absolute newest date boundary encountered
        if blog_datetime > newest_timestamp:
            newest_timestamp = blog_datetime

    # Process all text-based items collectively if any were bundled
    if raw_text_docs:
        chunks = rag.ingest_raw_documents(raw_text_docs)
        total_chunks += chunks

    # 4. Save and commit index vector adjustments if changes occurred
    if total_chunks > 0:
        _save_and_sync(rag)

    # 5. Flush state progress timestamp to Firestore
    if project:
        try:
            db_fs = _firestore_client()
            db_fs.collection("system_config").document("blog_sync").set(
                {
                    "last_sync_time": newest_timestamp.isoformat(),
                    "execution_ran_at": datetime.now(timezone.utc).isoformat()
                }, 
                merge=True
            )
        except Exception as exc:
            logger.warning("blogs_sync: Firestore checkpoint save failed: %s", exc)

    return {
        "synced": len(processed_titles),
        "chunks_stored": total_chunks,
        "processed_items": processed_titles
    }