# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Behavioral Guidelines

**Think before coding.** State assumptions explicitly. If multiple interpretations exist, present them — don't pick silently. Push back when a simpler approach exists.

**Minimum code that solves the problem.** No features beyond what was asked. No abstractions for single-use code. If you write 200 lines and it could be 50, rewrite it.

**Surgical changes.** Touch only what you must. Don't improve adjacent code, comments, or formatting. Match existing style. Remove only the imports/variables that YOUR changes made unused — not pre-existing dead code.

---

## Running Locally

```bash
pip install -r requirements.txt
playwright install chromium
uvicorn server:app --reload --port 8000
```

- Open `http://localhost:8000/admin` for the local ingest UI
- Auth is **bypassed entirely** when `GOOGLE_CLOUD_PROJECT` is unset — no password needed locally
- FAISS index persists to `./rag_index/`, scraper cache to `./scraper_cache/`

`.env` for local dev (prod vars left blank = local mode):
```
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
YOUTUBE_API_KEY=...
GOOGLE_CLOUD_PROJECT=     # leave blank for local dev
GCS_BUCKET=               # leave blank for local dev
```

---

## Architecture

This is a RAG chatbot for **Dr. Swati Desai** (2Meditate). Users chat with an AI avatar grounded in her content (website, PDFs, YouTube videos).

```
frontend/index.html       — Public chat UI (Vercel, no auth)
frontend-admin/index.html — Admin panel (Vercel, password-protected)
        │
        ▼
server.py (FastAPI on Cloud Run, gunicorn + UvicornWorker, 2 workers)
        │
        ├── rag_query.py       — Claude tool-use RAG + session management
        ├── orchestrator.py    — Ingestion coordinator
        │       ├── scraper.py     — Playwright crawl, Gemini clean, YouTube, PDFs
        │       ├── chunker.py     — Text splitting + metadata headers
        │       └── database.py   — FAISS + BM25 hybrid search
        │
        ├── firestore_sessions.py — Session store (Firestore prod / in-memory local)
        └── GCS / Firestore       — Production persistence
```

**Production mode is triggered by `GOOGLE_CLOUD_PROJECT` being set.** When set:
- Sessions stored in Firestore (`FIRESTORE_DB` database, `rag_sessions` collection)
- Admin auth enforced (Argon2 hash stored in Firestore `system_config/admin`)
- FAISS index downloaded from GCS on startup, uploaded after every ingest

**Local mode** (env var unset): in-memory sessions, local FAISS, no auth.

---

## Key Files

**`server.py`** — All API routes. Streaming works by running the synchronous Anthropic SDK in a `ThreadPoolExecutor` and feeding tokens into a `queue.Queue` that the async SSE handler polls.

**`rag_query.py`** — Claude is given one tool: `search_portfolio` (calls FAISS hybrid search). Multi-turn history is injected via the session store. `stream_answer()` yields text tokens; the final `GeminiAnswer` comes back via `StopIteration.value`.

**`database.py`** — Gemini `gemini-embedding-001` (3072-dim), `faiss.IndexFlatIP` with L2-normalized vectors (= cosine similarity), BM25Okapi. Hybrid blend: **60% dense + 40% sparse**.

**`orchestrator.py`** — All ingestion paths go through `_store_docs()`: corruption guard → chunk → optional embedding dedup → upsert (deletes old chunks for same URL first) → remove short chunks.

**`scraper.py`** — Playwright headless Chromium for websites; Gemini Files API for PDFs/Office; YouTube Data API v3 + transcript fallback to Gemini native video URL. `ScraperCache` in `./scraper_cache/` lets you rebuild the FAISS index with zero API calls.

**`cloudbuild.yaml`** — CI/CD for Cloud Run. Sets `GOOGLE_CLOUD_PROJECT`, `GCS_BUCKET`, `FIRESTORE_DB` as env vars. API keys (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `YOUTUBE_API_KEY`) must be added manually in the Cloud Run console — they survive redeploys.

---

## Admin Auth Flow

1. Frontend calls `GET /admin/setup/status`
2. If unconfigured → setup form shown → `POST /admin/setup` stores Argon2 hash in Firestore
3. Login stores plaintext password in `sessionStorage` as Bearer token
4. Every admin request sends `Authorization: Bearer <password>`; server runs `password_hasher.verify()` per request

---

## Deployment (Cloud Run)

Cloud Build trigger on `main` branch push — reads `cloudbuild.yaml`. Substitution values in `cloudbuild.yaml`:
- `_SERVICE_NAME`: `swati-avatar`
- `_PROJECT_ID`: `gen-lang-client-0966906205`
- `_GCS_BUCKET`: `swati-rag-index`
- `_FIRESTORE_DB`: `swati-chats`
- `_REGION`: `asia-south1`

Cloud Run service account needs: `roles/datastore.user` + `roles/storage.objectAdmin` on the bucket.

---

## Standalone CLI Tools (not part of main pipeline)

```bash
python YoutubeScraper.py --handle @ChannelHandle --max 50  # pre-scrape YouTube
python delete.py                                            # interactive index management
```
