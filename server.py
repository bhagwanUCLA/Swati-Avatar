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
import queue as _sync_queue
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

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

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO)
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
    2. Download the FAISS index from GCS (if configured).
    """
    global _cached_admin_hash
    
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
    import requests as _req

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
    List files in the shared OneDrive folder, ingest any added/modified
    since the last sync, and update the last_sync_time in Firestore.
    Called weekly by Cloud Scheduler.
    """
    import base64
    import requests as _req

    share_url = os.environ.get("ONEDRIVE_SHARE_URL", "")
    if not share_url:
        return {"synced": 0, "warning": "ONEDRIVE_SHARE_URL not set"}

    encoded = base64.urlsafe_b64encode(("u!" + share_url).encode()).decode().rstrip("=")
    try:
        resp = _req.get(
            f"https://graph.microsoft.com/v1.0/shares/{encoded}/driveItem/children",
            params={"$select": "id,name,lastModifiedDateTime,file,@microsoft.graph.downloadUrl"},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.error("onedrive_sync: Graph API error: %s", exc)
        raise HTTPException(status_code=502, detail=f"OneDrive API error: {exc}")

    all_files = [item for item in resp.json().get("value", []) if "file" in item]

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
        return {"synced": 0, "message": "No new files since last sync"}

    rag = _get_rag()
    total_chunks = 0
    synced_names = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for item in new_files:
            download_url = item.get("@microsoft.graph.downloadUrl")
            if not download_url:
                logger.warning("onedrive_sync: no download URL for %s", item["name"])
                continue
            try:
                file_bytes = _req.get(download_url, timeout=60).content
                dest = os.path.join(tmpdir, item["name"])
                Path(dest).write_bytes(file_bytes)
                synced_names.append(item["name"])
                logger.info("onedrive_sync: downloaded %s", item["name"])
            except Exception as exc:
                logger.warning("onedrive_sync: failed to download %s: %s", item["name"], exc)

        if synced_names:
            total_chunks = rag.ingest_folder(tmpdir, section="onedrive", recursive=False)

    if total_chunks > 0:
        _save_and_sync(rag)

    if project:
        try:
            db_fs = _firestore_client()
            db_fs.collection("system_config").document("onedrive_sync").set(
                {"last_sync_time": datetime.now(timezone.utc).isoformat()}, merge=True
            )
        except Exception as exc:
            logger.warning("onedrive_sync: Firestore write failed: %s", exc)

    return {"synced": len(synced_names), "chunks_stored": total_chunks, "files": synced_names}