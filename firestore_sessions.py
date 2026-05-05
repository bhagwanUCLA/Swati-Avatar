"""
firestore_sessions.py
---------------------
Session stores for Claude conversation history.

Two implementations with an identical interface:

  FirestoreSessionStore  — backed by Google Cloud Firestore (production).
                           Requires GOOGLE_CLOUD_PROJECT env var and
                           the google-cloud-firestore package.

  InMemorySessionStore   — plain dict (local development / fallback).
                           History is lost on process restart.

Both are used by rag_query.RAG via its `session_store` constructor argument.
server.py picks the right one at startup based on GOOGLE_CLOUD_PROJECT.

Firestore document layout
--------------------------
Collection : rag_sessions
Document ID: <session_id>  (e.g. "browser-uuid-1234")
Fields:
  history    : list[dict]   — Claude message history
  updated_at : Timestamp    — server-side, set on every write
  created_at : Timestamp    — set only on first write (merge=True)
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol  (structural interface — both stores satisfy it)
# ---------------------------------------------------------------------------

class SessionStore(Protocol):
    def get(self, session_id: str) -> list[dict]: ...
    def save(self, session_id: str, history: list[dict]) -> None: ...
    def delete(self, session_id: str) -> None: ...
    def list_all(self) -> list[str]: ...


# ---------------------------------------------------------------------------
# In-memory store  (local dev / fallback)
# ---------------------------------------------------------------------------

class InMemorySessionStore:
    """
    Simple dict-backed session store.

    Thread-safe for the typical single-process dev server.
    Not safe across gunicorn workers — use FirestoreSessionStore in prod.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[dict]] = {}

    def get(self, session_id: str) -> list[dict]:
        return list(self._store.get(session_id, []))

    def save(self, session_id: str, history: list[dict]) -> None:
        self._store[session_id] = list(history)

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def list_all(self) -> list[str]:
        return list(self._store.keys())

    def list_paginated(self, limit: int = 20, offset: int = 0) -> dict:
        items = list(self._store.items())
        total = len(items)
        page = items[offset:offset + limit]
        return {
            "sessions": [
                {"session_id": sid, "updated_at": None, "history": list(hist)}
                for sid, hist in page
            ],
            "total": total,
        }


# ---------------------------------------------------------------------------
# Firestore store  (production on GCP)
# ---------------------------------------------------------------------------

class FirestoreSessionStore:
    """
    Firestore-backed session store.

    Each session maps to one document in the `rag_sessions` collection.
    The `history` field stores the full list of Claude message dicts.

    IAM requirements
    ----------------
    The Cloud Run service account must have:
      roles/datastore.user   (Firestore read + write)

    Parameters
    ----------
    project : GCP project ID (e.g. "gen-lang-client-0368862224")
    collection : Firestore collection name (default: "rag_sessions")
    """

    COLLECTION = "rag_sessions"

    def __init__(self, project: str, collection: str = "rag_sessions") -> None:
        from google.cloud import firestore  # lazy import — not needed in local dev
        database = os.environ.get("FIRESTORE_DB", "(default)")
        self._db = firestore.Client(project=project, database=database)
        self.COLLECTION = collection
        logger.info(
            "FirestoreSessionStore ready: project=%s database=%s collection=%s",
            project, database, collection,
        )

    def _doc_ref(self, session_id: str):
        return self._db.collection(self.COLLECTION).document(session_id)

    def get(self, session_id: str) -> list[dict]:
        """Load a session's history.  Returns [] if not found."""
        try:
            snap = self._doc_ref(session_id).get()
            if snap.exists:
                data = snap.to_dict() or {}
                return list(data.get("history", []))
            return []
        except Exception as exc:
            logger.error("Firestore get(%s) failed: %s", session_id, exc)
            return []

    def save(self, session_id: str, history: list[dict]) -> None:
        """Upsert a session's full history.  Uses merge=True so other fields survive."""
        from google.cloud import firestore
        try:
            self._doc_ref(session_id).set(
                {
                    "history":    history,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
        except Exception as exc:
            logger.error("Firestore save(%s) failed: %s", session_id, exc)

    def delete(self, session_id: str) -> None:
        """Delete a session document entirely."""
        try:
            self._doc_ref(session_id).delete()
        except Exception as exc:
            logger.error("Firestore delete(%s) failed: %s", session_id, exc)

    def list_all(self) -> list[str]:
        """Return all session IDs in the collection."""
        try:
            return [doc.id for doc in self._db.collection(self.COLLECTION).stream()]
        except Exception as exc:
            logger.error("Firestore list_all() failed: %s", exc)
            return []

    def list_paginated(self, limit: int = 20, offset: int = 0) -> dict:
        """Return paginated sessions ordered by most recent (updated_at desc)."""
        from google.cloud import firestore as _fs
        try:
            col = self._db.collection(self.COLLECTION)
            total = len([doc.id for doc in col.select([]).stream()])
            query = (
                col.order_by("updated_at", direction=_fs.Query.DESCENDING)
                .offset(offset)
                .limit(limit)
            )
            sessions = []
            for doc in query.stream():
                data = doc.to_dict() or {}
                updated = data.get("updated_at")
                updated_iso = updated.isoformat() if hasattr(updated, "isoformat") else None
                sessions.append({
                    "session_id": doc.id,
                    "updated_at": updated_iso,
                    "history": data.get("history", []),
                })
            return {"sessions": sessions, "total": total}
        except Exception as exc:
            logger.error("Firestore list_paginated() failed: %s", exc)
            return {"sessions": [], "total": 0}
