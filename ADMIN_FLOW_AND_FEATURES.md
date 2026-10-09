# Swati RAG Admin Flow and Features

## Purpose

The admin panel manages the system prompt, chat history, the shared knowledge base,
manual content ingestion, and scheduled ingestion activity. It is a single-page
application served from `frontend-admin/index.html` and talks to the FastAPI backend
in `server.py`.

## Authentication

1. The panel calls `GET /admin/setup/status` when it loads.
2. If no admin password has been configured, it shows the setup form and calls
   `POST /admin/setup`.
3. Otherwise the admin signs in through `POST /login`.
4. The backend returns a JWT access token. The browser keeps it in
   `sessionStorage` and sends it as `Authorization: Bearer <token>` on admin requests.
5. A `401` clears the token and returns the panel to the login screen.

In production, the Argon2 password hash and chat/session data are stored in
Firestore. Local development can bypass Firestore authentication when no Google Cloud
project is configured.

## Admin Tabs

The tabs appear in this order:

1. **Prompt** — view, reload, and save the avatar system prompt.
2. **Chats** — view persisted conversation history.
3. **Knowledge Base** — inspect the current FAISS index, browse documents, and delete
   an indexed document.
4. **Paste Text** — queue manually pasted text for indexing.
5. **File / Zip** — queue one supported file or every supported entry in a ZIP file.
6. **Videos** — queue YouTube video or playlist URLs submitted by an admin.
7. **Scheduled Jobs** — the single, newest-first queue and activity log for every
   ingestion source.
8. **Cleanup** — preview and apply chunk-cleanup filters.

The model selector above the tabs is available to authenticated admins. It reads and
writes the selected chat model through the admin model endpoints.

## Shared Knowledge Base Storage

There is one shared RAG index for every ingestion source.

| Environment | FAISS index | Metadata |
|---|---|---|
| Local | `rag_index/faiss.index` | `rag_index/metadata.pkl` |
| Production GCS | `rag_index/faiss.index` | `rag_index/metadata.pkl` |

On startup, production downloads this pair from the configured GCS bucket. After a
successful ingestion, cleanup, or document deletion, the backend saves the pair and
uploads both files back to the same primary GCS paths. Index writes are serialized with
one shared lock so the background worker and admin operations cannot write FAISS at the
same time.

Normal ingestion does **not** create or read release snapshots or a release manifest.

## One Shared Scheduled Queue and Activity Log

All ingestion work uses the same Firestore-backed worker queue. There are not separate
Drive, video, ZIP, blog, or section-specific queues.

Each submitted item is stored under a `sync_jobs` job document with an item record in
its `items` subcollection. The job records are used internally for durable worker
control, but the Scheduled Jobs tab displays one flat list of items across every job.
The list is ordered by item creation time, newest first.

Each Scheduled Jobs row shows:

- **Item** — file name, pasted-text title, blog title, or submitted video URL.
- **Source** — `paste`, `files`, `videos`, `gdrive`, `blogs`, or `youtube`.
- **Status** — `pending`, `processing`, `completed`, or `failed`.
- **Attempts** — how many times the worker has claimed the item.
- **Remove** — available only for pending or failed items.

### Item Lifecycle

```text
pending → processing → completed
                    └→ pending (automatic retry while attempts remain)
                    └→ failed  (after the retry limit)
```

Completed and failed records remain in Firestore as activity history. Only `pending`
items are eligible for processing. The worker processes one item at a time.

The backend has job-level pause and restart APIs for operational recovery:

- `POST /sync-jobs/{job_id}/pause` keeps completed items but resets every unfinished
  item (`pending`, `processing`, or `failed`) to `pending` and pauses the job.
- `POST /sync-jobs/{job_id}/restart` resumes a paused job and wakes the worker.

The flat Scheduled Jobs interface intentionally does not group or display items by job.

### Queue APIs

| Endpoint | Purpose |
|---|---|
| `GET /sync-items?limit=500` | Newest-first flat list used by Scheduled Jobs. |
| `GET /sync-jobs` | Internal/recovery-oriented list of job records. |
| `GET /sync-jobs/{job_id}` | One job and its item records. |
| `POST /sync-jobs/{job_id}/pause` | Pause and reset unfinished items. |
| `POST /sync-jobs/{job_id}/restart` | Resume a paused job. |
| `DELETE /sync-jobs/{job_id}/items/{item_id}` | Remove a pending or failed item. |

## Manual Ingestion

### Paste Text

The panel cleans pasted text in the browser, then posts it to `POST /ingest/documents`.
The endpoint creates `paste` items and wakes the shared worker. It returns immediately
with a queue result; embedding and FAISS writes happen later in the worker.

Each item retains its title, selected section, optional source URL, document type, and
text content in its Firestore metadata. When claimed, the worker chunks and embeds the
text, updates the shared primary index pair, and marks the item completed.

### File / Zip

`POST /ingest/folder` accepts one supported file or a ZIP archive. A ZIP is inspected
without extracting it to arbitrary server paths. Every supported non-directory,
non-symlink entry becomes an individual `files` queue item, so each entry is visible in
Scheduled Jobs.

Because the worker runs after the upload request finishes, the uploaded bytes are staged
temporarily in the configured GCS bucket under `sync_uploads/`. This is temporary input
storage, not another FAISS index or metadata store. The worker downloads one staged
file, ingests it with its selected section, writes the shared primary index pair, and
removes the staged object after a successful completion. Failed items retain their
staged input for retry.

Supported files use the existing orchestrator pipeline:

- Text: `.txt`, `.md`, `.markdown`, `.rst`, `.html`, `.htm`
- Documents/data: `.pdf`, `.docx`, `.doc`, `.odt`, `.pptx`, `.ppt`, `.xlsx`, `.xls`,
  `.xlsm`, `.xlsb`, `.csv`, `.ods`
- Media: `.mp4`, `.mov`, `.avi`, `.webm`, `.wmv`, `.mpg`, `.mpeg`, `.flv`, `.3gp`,
  `.3gpp`, and other extensions supported by the configured orchestrator

### Admin Videos

`POST /ingest/videos` turns each submitted YouTube video or playlist URL into a
`videos` queue item. The worker invokes the existing video/playlist ingestion pipeline
for one submitted URL at a time, then persists the shared index pair. Submitted videos
are therefore visible in Scheduled Jobs before processing completes.

## Automatic Ingestion Sources

These sources use the same item model and the same one-at-a-time worker as manual
ingestion.

### Google Drive

`POST /gdrive/sync` discovers supported new Drive files after the stored checkpoint,
creates `gdrive` items, persists the newest discovery time, wakes the worker, and
returns. The worker later downloads each file through the Drive API and ingests it with
a stable `gdrive://<file_id>` source URL.

### Blog CMS

`POST /blogs/sync` discovers new or updated CMS entries and creates `blogs` items. An
item carries either the blog text or a PDF URL. The worker indexes one blog item at a
time with a stable `cms://blog/<blog_id>` source URL.

### YouTube Notifications

`POST /youtube/notify` verifies the optional Hub signature, parses YouTube notification
entries, creates `youtube` items using the video ID, wakes the worker, and returns
`204` without waiting for transcription or indexing. The worker processes each video
later using its canonical YouTube URL.

## Knowledge Base

The Knowledge Base tab calls:

- `GET /stats` for chunk, document, section, embedding, and cache statistics.
- `GET /documents` for the document list.
- `GET /documents/{doc_index}/chunks` when an admin opens a document card.
- `DELETE /documents/{title}` to delete all chunks for a named document.

Document deletion and cleanup are serialized with worker writes and save the same shared
primary FAISS and metadata pair.

## Other Admin Features

### System Prompt

`GET /admin/system-prompt` loads the current prompt and
`POST /admin/system-prompt` saves it. The prompt is stored locally for the running
instance and synchronized to the configured GCS prompt object in production.

### Chats

`GET /sessions/history` returns paginated persisted chat sessions. The tab renders each
session as an expandable conversation. `DELETE /sessions` clears all sessions.

### Cleanup

`POST /cleanup/preview` performs a dry run for repeated-word, short-chunk, and regex
filters. `POST /cleanup/apply` permanently removes the confirmed matching chunks and
saves the primary index pair.

## Operational Notes

- The current deployment configuration runs one Gunicorn worker, one Cloud Run instance,
  and one request at a time so the in-process queue worker has a single writer.
- If the persisted primary index cannot be downloaded on startup, index writes and queue
  processing are blocked rather than risking an empty or divergent FAISS index.
- A failed item records its error and remains visible. Interrupted processing is
  recovered after its processing lease expires.
- The panel uses “Queued” messages for Paste Text, File / Zip, and Videos because the
  HTTP request creates durable work; it does not wait for embedding or FAISS writes.
