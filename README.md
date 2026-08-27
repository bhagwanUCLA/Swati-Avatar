# 2Meditate RAG Chatbot — Project Documentation

A Retrieval-Augmented Generation (RAG) system for **Dr. Swati Desai** (2Meditate). Users chat with an AI assistant that answers questions grounded in her mindfulness teachings, psychological insights, and content from her website, YouTube, PDFs, and ingested documents.

---

## Quick Start

```bash
# 1. Install + cache playwright
pip install -r requirements.txt
playwright install chromium

# 2. Create .env with API keys
echo "GEMINI_API_KEY=your_key" > .env
echo "ANTHROPIC_API_KEY=your_key" >> .env
echo "YOUTUBE_API_KEY=optional" >> .env

# 3. Run server (no auth locally)
uvicorn server:app --reload --port 8000

# 4. Open browser
# Chat: http://localhost:8000
# Admin: http://localhost:8000/admin
```

**Local mode**: No Firestore needed, no password required, FAISS stored in `./rag_index/`.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  PUBLIC FRONTEND (Vercel)          ADMIN FRONTEND (Vercel)          │
│  Chat UI — no auth                 frontend-admin/index.html        │
│  /query/stream  (SSE)              Password-protected admin panel   │
└───────────────────┬─────────────────────────────┬───────────────────┘
                    │ HTTPS                        │ HTTPS + Bearer token
                    ▼                              ▼
        ┌───────────────────────────────────────────────┐
        │              server.py  (FastAPI)              │
        │  Cloud Run · gunicorn + UvicornWorker          │
        │                                                │
        │  Public routes:  /query/stream, /query,        │
        │                  /health, /stats               │
        │  Admin routes:   /ingest/*, /documents,        │
        │                  /cleanup/*, /sessions/*,      │
        │                  /config, /index               │
        │  Auth routes:    /admin/setup, /admin/setup/status │
        └───────────┬────────────────────┬──────────────┘
                    │                    │
          ┌─────────▼──────┐   ┌────────▼────────────┐
          │  orchestrator  │   │     rag_query.py     │
          │  RAGOrchestrator│   │  RAG (Claude tool-  │
          │  Ingestion coord│   │  use + session mgmt)│
          └─────────┬──────┘   └────────┬────────────┘
                    │                    │
       ┌────────────┼──────┐   ┌────────▼────────────┐
       │            │      │   │  Anthropic Claude    │
       ▼            ▼      ▼   │  (claude-sonnet-4-6) │
  scraper.py   chunker.py  │   └─────────────────────┘
  PortfolioScraper │       │
  │ WebsiteCrawler │       │
  │ GeminiCleaner  │       │
  │ YT scraper     ▼       ▼
  │            database.py (FAISSDatabase)
  │            ├─ Gemini embedding (gemini-embedding-001)
  │            ├─ FAISS IndexFlatIP (cosine similarity)
  │            └─ BM25Okapi (hybrid: 60% dense + 40% sparse)
  │
  ├─ ./scraper_cache/     (disk cache — pages + video summaries)
  │
  ├─ ./rag_index/         (local FAISS index — dev only)
  │   ├─ faiss.index
  │   └─ metadata.pkl
  │
  └─ GCS bucket           (FAISS index in production)
      ├─ rag_index/faiss.index
      └─ rag_index/metadata.pkl

Firestore (production persistence)
  ├─ rag_sessions/{session_id}   — chat histories
  └─ system_config/admin         — Argon2 password hash
```

---

## Components

### `server.py` — FastAPI Backend
- **50+ endpoints**: Public chat, admin ingestion, auth, YouTube webhooks, session/document management.
- **Admin auth**: Bearer token validated against Argon2 hash in Firestore (production). Auth bypassed in local dev when `GOOGLE_CLOUD_PROJECT` unset.
- **Streaming architecture**: Anthropic SDK is blocking; `_run_llm_in_thread()` runs it in a `ThreadPoolExecutor`, feeding tokens into a thread-safe `queue.Queue`. Async `event_stream()` polls queue with short sleeps, keeping event loop responsive for other requests.
- **GCS persistence**: FAISS index downloaded on startup, uploaded after every ingest.
- **Session store**: Firestore (production) or in-memory dict (local dev).
- **YouTube webhooks**: PubSubHubbub push notifications trigger auto-ingestion of new videos.

### `orchestrator.py` — RAGOrchestrator
Central ingestion coordinator. Routes all content (websites, YouTube, files, raw text) through a unified pipeline:
- **Corruption guard**: Detects binary junk (excessive control characters) and skips corrupt content
- **Chunking**: Splits documents into 3500-token chunks with 50-token overlap
- **Deduplication**: Optional O(n²) cosine-similarity check before indexing
- **Upsert semantics**: Deletes old chunks for the same URL before storing new ones (prevents duplicates on re-ingest)
- **Quality filtering**: Removes chunks below minimum token count and applies optional regex/keyword filters

### `scraper.py` — PortfolioScraper
Unified content extraction engine for all source types:
- **WebsiteCrawler**: Playwright headless Chromium crawl with JS execution (expands accordions), strips navigation/footer/scripts, extracts clean Markdown
- **GeminiCleaner**: Sends raw crawled Markdown to Gemini API for semantic cleaning (removes boilerplate, formats lists/tables)
- **YouTubeChannelScraper**: YouTube Data API v3 for channel metadata + video list, pulls transcripts via `YouTubeTranscriptApi`, falls back to Gemini's native video URL analysis if transcripts unavailable
- **File extraction (`_file_stage3_gemini()`)**: Uploads PDF/Word/PowerPoint/Excel to Gemini Files API with type-specific extraction prompts
- **ScraperCache**: Disk cache at `./scraper_cache/` storing raw pages and video summaries. **Critical feature**: allows rebuilding the entire FAISS index with zero API calls (replay from cache)

### `rag_query.py` — RAG (Claude Tool-Use + Multi-Turn)
- **Wraps FAISSDatabase + Claude**: Retrieval engine + LLM generation combined.
- **Tool-based retrieval**: Claude is given a single tool—`search_portfolio`—which calls FAISS hybrid search. Claude decides when/how to search for follow-up context.
- **Impersonation**: System prompt instructs Claude to embody Dr. Swati Desai's voice (warm, integrative, mindfulness-focused).
- **Session management**: Multi-turn chat history injected from session store (Firestore/in-memory).
- **Streaming**: `stream_answer()` is a sync generator yielding tokens in real-time. Returns `GeminiAnswer` dataclass (answer text, sources list, token count) via `StopIteration.value`.
- **Source tracking**: Each retrieved chunk includes title, section, URL, doc type, relevance score.

### `database.py` — FAISSDatabase (Hybrid Search)
- **Embedding**: Google Gemini `gemini-embedding-001` (3072-dim). `RETRIEVAL_DOCUMENT` task for indexing, `RETRIEVAL_QUERY` for search queries.
- **Dense index**: `faiss.IndexFlatIP` with L2-normalized vectors = cosine similarity (semantic search).
- **Sparse index**: BM25Okapi for keyword-based retrieval.
- **Hybrid blend**: Retrieves top-k from both indexes, min-max normalizes scores, then combines: **60% dense + 40% sparse** for final ranking.
- **Persistence**: `faiss.index` (binary FAISS) + `metadata.pkl` (chunk metadata dict).

### `chunker.py` — DocumentChunker
- Uses `langchain.RecursiveCharacterTextSplitter` (default chunk_size=3500, overlap=50).
- Injects a metadata header into each chunk: `## Chunk N | section | title` (embedded alongside content).
- Optional O(n²) cosine-similarity deduplication before indexing.

### `firestore_sessions.py` — Session Stores
- `FirestoreSessionStore`: production, backed by Firestore collection `rag_sessions`.
- `InMemorySessionStore`: local dev fallback, plain dict (lost on restart).
- Both implement the same `SessionStore` protocol (`get`, `save`, `delete`, `list_all`).

### `YoutubeScraper.py` — Standalone YouTube CLI
Independent tool for scraping YouTube channels/videos to `./videos/{video_id}.json`. **Not used by the main pipeline.** Use it to pre-scrape a channel; the results can then be ingested via `/ingest/documents`.

```bash
python YoutubeScraper.py --handle @ChannelName --max 50
python YoutubeScraper.py --video VIDEO_ID
```

### `ProfileScraper.py` — Standalone Profile CLI
Independent tool for scraping LinkedIn-style academic profiles with Gemini-powered cleaning. **Not used by the main pipeline.**

### `delete.py` — Standalone Index CLI
Interactive CLI for index management: delete by URL/title/section, quality filtering (repeated words, short chunks, regex), skip-list management, cache auditing.

```bash
python delete.py
```

---

## Ingestion Pipeline Walkthrough

```
                     INPUT SOURCES
         ┌──────────┬──────────┬──────────┐
         ▼          ▼          ▼          ▼
      Website   YouTube    Files    Raw Text
         │          │          │          │
         └──────────┴──────────┴──────────┘
                    │
                    ▼
  server.py routes (admin-protected):
    POST /ingest          → crawl portfolio URL(s)
    POST /ingest/folder   → upload ZIP, PDF, Office files
    POST /ingest/videos   → YouTube URLs or playlists
    POST /ingest/documents→ paste raw text directly
                    │
                    ▼
  PortfolioScraper.scrape() — content extraction:
    Website:  Playwright headless crawl → expand JS → strip nav/footer
              → Gemini "clean this page" → Markdown
    YouTube:  Data API v3 metadata → transcript (or Gemini video URL)
    File:     Gemini Files API → extract text (type-aware: PDF, Word, etc.)
    Raw text: pass through as-is
                    │
                    ▼
  orchestrator._store_docs() — ingestion pipeline:
    1. Corruption guard (% control chars > threshold? → skip)
    2. DocumentChunker.chunk() → 3500-token chunks (50-token overlap)
    3. [Optional] dedupe by embedding cosine similarity
    4. Delete old chunks for same URL (upsert semantics)
    5. FAISSDatabase.add() → embed via Gemini + store in FAISS + BM25
    6. Remove short chunks (< min_tokens)
    7. Save to disk + upload to GCS (production)
                    │
                    ▼
             Index updated ✓
```

---

## Query Pipeline (Chat Flow)

When a user asks a question via `/query/stream` or `/query`:

```
1. SESSION LOOKUP
   └─ Retrieve chat history from Firestore/in-memory store

2. HYBRID SEARCH (Claude via tool-use)
   └─ Query embedded via Gemini API (3072-dim)
   └─ Search FAISS index (semantic): top-k results
   └─ Search BM25 index (keyword): top-k results
   └─ Blend scores (60% dense + 40% sparse)
   └─ Return top-n chunks with relevance scores

3. CLAUDE GENERATION
   └─ System prompt: impersonate Dr. Swati
   └─ Context: last 10 messages (multi-turn history)
   └─ Retrieved chunks injected into context
   └─ Claude calls search_portfolio tool if needs more context
   └─ Generate response with streaming tokens

4. RESPONSE + SOURCES
   └─ Stream tokens via SSE (Server-Sent Events)
   └─ Include chunk metadata: title, section, URL, score
   └─ Save to session store for next turn
```

---

## API Endpoints

### Public (no auth)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/stats` | Index + cache statistics |
| GET | `/query/stream` | SSE streaming chat (`question`, `session_id`, filters) |
| POST | `/query` | Single-shot (blocking) chat |
| GET | `/admin/setup/status` | Check if admin password is configured |
| POST | `/admin/setup` | Set initial admin password (one-time) |
| GET | `/admin` | Admin UI (serves `ingest_ui.html` locally; JSON redirect in production) |
| GET | `/youtube/notify` | YouTube PubSubHubbub verification (echoes `hub.challenge`) |
| POST | `/youtube/notify` | YouTube push webhook — auto-ingests new videos |

### Admin (Bearer token required in production)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/config` | Current server config (API keys masked) |
| POST | `/config` | Update config (resets RAG instance) |
| POST | `/ingest` | Crawl portfolio URL |
| POST | `/ingest/folder` | Upload ZIP or PDF file |
| POST | `/ingest/documents` | Inject raw text documents |
| POST | `/ingest/videos` | Ingest YouTube URLs / playlists |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{title}` | Delete all chunks for a document |
| POST | `/cleanup/preview` | Dry-run quality filter (repeated words, short chunks, regex) |
| POST | `/cleanup/apply` | Apply quality filter and delete matched chunks |
| GET | `/sessions` | List session IDs |
| GET | `/sessions/history` | All sessions with full message history |
| DELETE | `/sessions/{id}` | Clear one session |
| DELETE | `/sessions` | Clear all sessions |
| DELETE | `/index` | Wipe entire FAISS index |
| POST | `/youtube/resubscribe` | Re-register channels with YouTube PubSubHubbub |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key (embeddings + content extraction) |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key (Claude for chat) |
| `YOUTUBE_API_KEY` | No | YouTube Data API v3 key (YouTube ingestion) |
| `GOOGLE_CLOUD_PROJECT` | Prod only | GCP project ID — enables Firestore + auth |
| `GCS_BUCKET` | Prod only | GCS bucket name for FAISS index persistence |
| `WATCHED_CHANNEL_IDS` | PubSubHubbub | Comma-separated YouTube channel IDs to watch for new videos |
| `PUBSUB_SECRET` | PubSubHubbub | HMAC secret for verifying YouTube push payloads (`openssl rand -hex 32`) |

Set these in a `.env` file for local dev (loaded automatically via `python-dotenv`).
`WATCHED_CHANNEL_IDS` and `PUBSUB_SECRET` should be set in the Cloud Run console for production.

---

## Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 2. Create .env
cat > .env <<EOF
GEMINI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
YOUTUBE_API_KEY=your_key   # optional
EOF

# 3. Start the server (auth is bypassed locally — no password needed)
uvicorn server:app --reload --port 8000

# 4. Open the local admin UI to ingest content
open http://localhost:8000/admin

# 5. (Optional) Use the standalone YouTube scraper
python YoutubeScraper.py --handle @ChannelHandle --max 50
```

The local FAISS index is saved to `./rag_index/`. The scraper cache is saved to `./scraper_cache/`.

---

## Production Deployment (Cloud Run)

### Prerequisites
- GCP project with Firestore (Native mode) enabled
- GCS bucket for FAISS index
- Cloud Run service account with:
  - `roles/datastore.user` (Firestore read/write)
  - `roles/storage.objectAdmin` on the GCS bucket

### Build & Deploy

```bash
# Build and push container
gcloud builds submit --config cloudbuild.yaml

# Or manually:
docker build -t gcr.io/PROJECT_ID/portfolio-rag .
docker push gcr.io/PROJECT_ID/portfolio-rag

gcloud run deploy portfolio-rag \
  --image gcr.io/PROJECT_ID/portfolio-rag \
  --platform managed \
  --region us-central1 \
  --set-env-vars GEMINI_API_KEY=...,ANTHROPIC_API_KEY=...,GOOGLE_CLOUD_PROJECT=...,GCS_BUCKET=...
```

### Gunicorn start command (in container)
```
gunicorn -k uvicorn.workers.UvicornWorker server:app \
  --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

### First-run setup
1. Deploy the service
2. Open the admin frontend URL
3. You will be prompted to set an admin password (stored as Argon2 hash in Firestore)
4. Log in and start ingesting content via the admin panel

---

## YouTube PubSubHubbub (Auto-Ingestion)

New videos published on watched channels are automatically transcribed and added to the FAISS index within ~30 seconds of upload — no polling, no manual action.

### How it works
1. YouTube pushes an Atom XML payload to `POST /youtube/notify` when a new video is published
2. The endpoint parses the video ID, calls `ingest_videos`, and saves to GCS
3. Subscriptions expire after 30 days — Cloud Scheduler re-registers every 15 days

### Setup (one-time)

**1. Set env vars in Cloud Run console:**

| Variable | Value |
|---|---|
| `WATCHED_CHANNEL_IDS` | YouTube channel ID(s), comma-separated (e.g. `UCxxxxxxxxxxxxxx`) |
| `PUBSUB_SECRET` | Random secret: `openssl rand -hex 32` |

**2. Bootstrap the subscription** — click "Resubscribe YouTube" in the admin panel, or:
```bash
curl -X POST https://<cloud-run-url>/youtube/resubscribe \
  -H "Authorization: Bearer <admin-password>"
```
YouTube calls `GET /youtube/notify` automatically to verify. Check logs for `PubSubHubbub subscribe: ... status=202`.

**3. Create Cloud Scheduler job** (GCP Console → Cloud Scheduler → Create Job):

| Field | Value |
|---|---|
| Frequency | `0 9 1,15 * *` (9am on 1st and 15th of every month) |
| Target | HTTP POST `https://<cloud-run-url>/youtube/resubscribe` |
| Header | `Authorization: Bearer <admin-password>` |

### Finding a YouTube channel ID
Go to the channel → View Page Source → search for `"channelId"`. Or use the YouTube Data API:
```
https://www.youtube.com/xml/feeds/videos.xml?channel_id=UCxxxxxxxxxxxxxx
```

---

## Frontends

| Frontend | Auth | Purpose | Deploy |
|----------|------|---------|--------|
| Public chat UI | None | End-user chat interface | Vercel |
| `frontend-admin/index.html` | Password (Bearer token) | Content management, ingestion, cleanup | Vercel |
| `ingest_ui.html` | None (local only) | Quick ingestion before cloud deployment | Served at `/admin` locally |

### Admin auth flow
1. On first load, frontend calls `GET /admin/setup/status`
2. If `is_set: false` → show password setup form → `POST /admin/setup` (stores Argon2 hash in Firestore)
3. On subsequent logins, user enters password → stored in `sessionStorage` as the Bearer token
4. Every admin API call sends `Authorization: Bearer <password>`
5. Server calls `password_hasher.verify(password, argon2_hash)` on each request

---

## Scraper Cache

The scraper cache is a disk-based key-value store in `./scraper_cache/`:

```
scraper_cache/
├─ pages/
│   └─ {md5(url)}.json    # { url, final_url, html/content, content_type, cached_at }
├─ videos/
│   └─ {md5(url)}.json    # { url, title, summary, cached_at }
└─ skip_list.json          # URLs never re-scraped (manually ingested pages)
```

The cache is **independent of the FAISS index**. You can rebuild the entire vector index from cache with zero API/network calls:

```
POST /ingest  { "url": "https://...", "rebuild": true }
```

---

## Key Design Decisions

### **Architecture & Performance**
- **Sync LLM in thread pool**: Anthropic SDK is blocking/synchronous. Running it in `ThreadPoolExecutor` prevents freezing the asyncio event loop on long queries—critical for scaling across concurrent users.
- **Gunicorn + UvicornWorker (2 workers)**: Parallel request handling; each worker has its own thread pool for LLM calls.
- **GCS ephemeral persistence**: Cloud Run containers are stateless/ephemeral. FAISS index downloaded on startup, uploaded after each ingest—survives restarts without redeployment.

### **Search & Retrieval**
- **Hybrid search (60% dense + 40% sparse)**: Combines semantic similarity (Gemini embeddings via FAISS cosine) with keyword matching (BM25Okapi). Handles edge cases: acronyms, titles, years, specific terms that keywords excel at.
- **Gemini embeddings (3072-dim)**: Higher dimensionality captures nuance; `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY` task types optimize for indexing vs. search respectively.

### **Data Integrity**
- **Upsert semantics**: Re-ingesting a URL deletes all old chunks for that URL first, preventing duplicates on update.
- **Corruption guard**: Binary junk detection (% control characters) prevents polluting index with corrupt PDFs or malformed files.
- **Scraper cache independence**: Cache (raw pages/videos) is separate from FAISS index—rebuild index with zero API calls by replaying cache.
- **Skip list**: Manually ingested raw documents are marked so portfolio crawls don't overwrite them.

### **User Experience**
- **Multi-turn sessions**: Chat history persisted per session ID; Claude sees last 10 messages for context.
- **Source attribution**: Every response includes chunk provenance (title, section, URL, relevance score).
- **Tool-based retrieval**: Claude decides *when* and *what* to search via `search_portfolio` tool—enabling follow-up research without user intervention.