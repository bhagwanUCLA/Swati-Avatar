# Portfolio RAG — Project Documentation

A Retrieval-Augmented Generation (RAG) system for the portfolio of **Bhagwan Chowdhry** (Finance Professor, ISB / UCLA). Users chat with an AI assistant that answers questions grounded in the professor's publications, videos, and web content.

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
- Serves both public and admin API endpoints
- **Admin auth**: Bearer token (plaintext password) verified against Argon2 hash stored in Firestore. Local dev bypasses auth entirely when `GOOGLE_CLOUD_PROJECT` is unset.
- **Streaming**: Anthropic SDK is synchronous; `_run_llm_in_thread()` runs it in a `ThreadPoolExecutor` and feeds tokens into a `queue.Queue`. The async `event_stream()` polls the queue, keeping the event loop free.
- **GCS sync**: Downloads FAISS index from GCS on startup; uploads after every ingest operation.
- **Session store**: Firestore in production, in-memory dict in local dev.

### `orchestrator.py` — RAGOrchestrator
Central coordinator. Wires together the scraper, chunker, and database. All ingestion paths converge at `_store_docs()`:
1. Filter corrupt content (binary junk check)
2. Chunk via `DocumentChunker`
3. Optional embedding-based deduplication
4. Upsert into `FAISSDatabase` (delete old chunks for same URL first)
5. Remove short chunks (`< min_tokens`)

### `scraper.py` — PortfolioScraper
Handles all web ingestion:
- **WebsiteCrawler**: Playwright headless Chromium. Expands accordions via JS, strips nav/footer/scripts.
- **GeminiCleaner**: Sends raw Markdown to Gemini API to produce clean profile documents.
- **YouTubeChannelScraper**: YouTube Data API v3 for metadata, `YouTubeTranscriptApi` for transcripts. Falls back to Gemini's native video URL feature if transcript is unavailable.
- **`_file_stage3_gemini()`**: Uploads PDF/Office files to Gemini Files API for extraction with type-specific prompts.
- **ScraperCache**: Disk cache in `./scraper_cache/` (pages + video summaries). Allows rebuilding the FAISS index with zero API calls.

### `rag_query.py` — RAG (Claude Tool-Use)
- Wraps `FAISSDatabase` for retrieval and Anthropic Claude for generation.
- Claude is given a single tool: `search_portfolio` (calls FAISS hybrid search).
- Impersonates Bhagwan Chowdhry via a system prompt.
- Maintains multi-turn chat history via the injected session store.
- `stream_answer()` yields text tokens; returns `GeminiAnswer` (answer, sources, token count) via `StopIteration.value`.

### `database.py` — FAISSDatabase
- **Embedding**: Google Gemini `gemini-embedding-001` (3072-dim). `RETRIEVAL_DOCUMENT` task type for indexing, `RETRIEVAL_QUERY` for search.
- **Index**: `faiss.IndexFlatIP` + `IndexIDMap` — inner product on L2-normalized vectors = cosine similarity.
- **Hybrid search**: BM25Okapi (sparse) combined with FAISS (dense). Scores are min-max normalized then blended: **60% dense + 40% sparse**.
- **Persistence**: `faiss.index` (binary) + `metadata.pkl` (pickled chunk dict).

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
URL / file / YouTube URL / raw text
         │
         ▼
  server.py  →  one of:
    POST /ingest             → orchestrator.ingest_portfolio(url)
    POST /ingest/folder      → orchestrator.ingest_folder(path)
    POST /ingest/videos      → orchestrator.ingest_videos(urls)
    POST /ingest/documents   → orchestrator.ingest_raw_documents(docs)
         │
         ▼
  PortfolioScraper (scraper.py)
    Website? → Playwright crawl → Gemini clean → ScrapedDocument list
    YouTube? → transcript / Gemini summary → ScrapedDocument list
    File?    → Gemini Files API / pypdf → ScrapedDocument list
         │
         ▼
  _store_docs (orchestrator.py)
    1. Corruption guard (binary junk → skip)
    2. DocumentChunker → DocumentChunk list
    3. [Optional] embedding dedup (cosine similarity)
    4. FAISSDatabase.add() — embed + upsert
    5. remove_short_chunks()
         │
         ▼
  save() + GCS upload
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

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key (embeddings + content extraction) |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key (Claude for chat) |
| `YOUTUBE_API_KEY` | No | YouTube Data API v3 key (YouTube ingestion) |
| `GOOGLE_CLOUD_PROJECT` | Prod only | GCP project ID — enables Firestore + auth |
| `GCS_BUCKET` | Prod only | GCS bucket name for FAISS index persistence |

Set these in a `.env` file for local dev (loaded automatically via `python-dotenv`).

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

- **Sync LLM in thread pool**: Anthropic SDK is blocking. Running it in `ThreadPoolExecutor` prevents freezing the asyncio event loop on long queries.
- **Hybrid search (60/40)**: Combines semantic similarity (dense) with keyword matching (sparse BM25) for better retrieval on specific terms like paper titles and years.
- **GCS for FAISS**: Cloud Run containers are ephemeral. FAISS index is downloaded from GCS on startup and uploaded after every ingest, so content survives container restarts without redeployment.
- **Upsert semantics**: Re-ingesting a URL deletes its old chunks first, preventing duplicates.
- **Skip list**: Manually ingested raw documents are added to the scraper skip list so a portfolio crawl doesn't overwrite them.