"""
database.py
-----------
FAISSDatabase — vector store backed by FAISS with Google Gemini embeddings
and BM25 keyword search (Hybrid Search).

Embedding backend
-----------------
Uses the Gemini Embedding API (gemini-embedding-001) via google-genai.
No local model is loaded — all embedding is done via API call.

Hybrid Search
-------------
Combines FAISS (Cosine Similarity, weight 0.8) and BM25 (weight 0.2).
Retrieves candidate pools from both dense and sparse indexes, min-max normalises 
their scores to a [0,1] scale, and combines them for the final ranking.
"""
from __future__ import annotations

import logging
import pickle
import re
import time
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from google import genai
from google.genai import types

from chunker import DocumentChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Safe: zero rows stay zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms

def _tokenize(text: str) -> list[str]:
    """Simple whitespace/punctuation tokenizer for BM25."""
    if not text:
        return []
    # Extract alphanumeric words
    return re.findall(r'(?u)\b\w\w+\b', text.lower())

# ---------------------------------------------------------------------------
# FAISSDatabase
# ---------------------------------------------------------------------------

class FAISSDatabase:
    """
    Hybrid Embedding + Sparse vector store for DocumentChunks.

    Parameters
    ----------
    model_name            Gemini embedding model. Default: "gemini-embedding-001".
    gemini_api_key        Google Gemini API key (required).
    index_path            Directory to load an existing index from on construction.
    batch_size            Texts per embed_content API call (default 100).
    output_dimensionality Vector size: 3072 (default), 1536, or 768.
    device                Ignored — kept for backward-compatible call sites.
    """

    _INDEX_FILE = "faiss.index"
    _META_FILE  = "metadata.pkl"

    def __init__(
        self,
        model_name:            str           = "gemini-embedding-001",
        gemini_api_key:        str           = "",
        index_path:            Optional[str] = None,
        batch_size:            int           = 100,
        output_dimensionality: int           = 3072,
        device:                str           = "cpu",   # unused, kept for compat
    ) -> None:
        self.model_name            = model_name
        self.batch_size            = batch_size
        self.output_dimensionality = output_dimensionality
        self._dim                  = output_dimensionality

        if not gemini_api_key:
            raise ValueError(
                "FAISSDatabase requires a Gemini API key.  "
                "Pass gemini_api_key= or set GEMINI_API_KEY in your environment."
            )

        self._client = genai.Client(api_key=gemini_api_key)

        logger.info(
            "FAISSDatabase ready: model=%s  dim=%d  batch_size=%d",
            model_name, self._dim, self.batch_size,
        )

        # FAISS: IndexFlatIP inside IndexIDMap (inner-product = cosine for L2-normed vecs)
        self._index: faiss.IndexIDMap = faiss.IndexIDMap(faiss.IndexFlatIP(self._dim))
        self._meta:  dict[int, DocumentChunk] = {}
        self._next_id: int = 0
        
        # BM25 State
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_id_map: list[int] = []

        if index_path and Path(index_path).exists():
            self.load(index_path)

    # -------------------------------------------------------------------------
    # BM25 Management
    # -------------------------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        """Rebuilds the BM25 index from the current metadata."""
        if not self._meta:
            self._bm25 = None
            self._bm25_id_map = []
            return

        corpus = []
        self._bm25_id_map = []
        for int_id, chunk in self._meta.items():
            self._bm25_id_map.append(int_id)
            # Use raw_content if available, fallback to text
            text_to_index = getattr(chunk, "raw_content", chunk.text)
            corpus.append(_tokenize(text_to_index))
        
        self._bm25 = BM25Okapi(corpus)

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    def _call_gemini_embed(
        self,
        texts:     list[str],
        task_type: str,
        label:     str = "embedding",
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")

        total     = len(texts)
        n_batches = (total + self.batch_size - 1) // self.batch_size
        all_vecs: list[np.ndarray] = []

        logger.info(
            "[embed:%s] starting — %d text(s)  %d batch(es)  task=%s",
            label, total, n_batches, task_type,
        )

        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end   = min(start + self.batch_size, total)
            batch = texts[start:end]

            logger.info(
                "[embed:%s] batch %d/%d  items %d–%d ...",
                label, batch_idx + 1, n_batches, start + 1, end,
            )
            t0 = time.perf_counter()

            result = self._client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.output_dimensionality,
                ),
            )

            elapsed = time.perf_counter() - t0
            vecs    = np.array([e.values for e in result.embeddings], dtype="float32")
            all_vecs.append(vecs)

            logger.info(
                "[embed:%s] batch %d/%d done  shape=%s  %.2fs",
                label, batch_idx + 1, n_batches, vecs.shape, elapsed,
            )

        combined = np.vstack(all_vecs)          # (total, dim)
        combined = _l2_normalise(combined)      # safety-normalise (always correct)

        logger.info(
            "[embed:%s] complete — %d vector(s)  dim=%d",
            label, len(combined), self._dim,
        )
        return combined

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._call_gemini_embed(texts, task_type="RETRIEVAL_DOCUMENT", label="doc")

    def embed_query(self, query: str) -> np.ndarray:
        logger.info("[embed:query] %r", query[:100])
        return self._call_gemini_embed([query], task_type="RETRIEVAL_QUERY", label="query")

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0].tolist()

    # -------------------------------------------------------------------------
    # Write operations
    # -------------------------------------------------------------------------

    def add(self, chunks: list[DocumentChunk]) -> None:
        existing = {c.chunk_id for c in self._meta.values()}
        new      = [c for c in chunks if c.chunk_id not in existing]
        if not new:
            logger.info("add() — nothing to do (%d chunk(s) already indexed).", len(chunks))
            return

        logger.info(
            "add() — embedding %d new chunk(s)  (skipping %d already present)",
            len(new), len(chunks) - len(new),
        )

        vecs    = self.embed([c.text for c in new])
        int_ids = np.arange(self._next_id, self._next_id + len(new), dtype=np.int64)
        self._index.add_with_ids(vecs, int_ids)

        for int_id, chunk in zip(int_ids, new):
            self._meta[int(int_id)] = chunk

        self._next_id += len(new)
        self._rebuild_bm25()
        logger.info("add() done — index now contains %d chunk(s).", len(self._meta))

    def delete_by_chunk_id(self, chunk_ids: list[str]) -> int:
        to_del = {iid for iid, c in self._meta.items() if c.chunk_id in chunk_ids}
        return self._remove_int_ids(list(to_del))

    def delete_by_doc_index(self, doc_index: int) -> int:
        to_del = [iid for iid, c in self._meta.items() if c.doc_index == doc_index]
        return self._remove_int_ids(to_del)

    def delete_by_doc_title(self, title: str) -> int:
        to_del = [iid for iid, c in self._meta.items() if c.doc_title == title]
        return self._remove_int_ids(to_del)

    def delete_by_section(self, section: str) -> int:
        to_del = [iid for iid, c in self._meta.items() if c.section == section]
        return self._remove_int_ids(to_del)

    def delete_by_url(self, url: str) -> int:
        to_del = [iid for iid, c in self._meta.items() if c.doc_url == url]
        return self._remove_int_ids(to_del)

    def delete_by_urls(self, urls: set[str]) -> int:
        to_del = [iid for iid, c in self._meta.items() if c.doc_url in urls]
        return self._remove_int_ids(to_del)

    def get_indexed_urls(self) -> set[str]:
        return {c.doc_url for c in self._meta.values() if c.doc_url}

    def clear(self) -> None:
        self._index.reset()
        self._meta.clear()
        self._next_id = 0
        self._rebuild_bm25()
        logger.info("Index cleared.")

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    def search(
        self,
        query:            str,
        top_k:            int           = 5,
        section_filter:   Optional[str] = None,
        doc_index_filter: Optional[int] = None,
    ) -> list[dict]:
        if not self._meta:
            logger.info("[search] index is empty — returning []")
            return []

        # Retrieve a broader candidate pool to ensure good hybrid overlap
        fetch_k = min(top_k * 5, len(self._meta))
        logger.info("[search] hybrid query=%r  top_k=%d  fetch_k=%d", query[:80], top_k, fetch_k)

        # 1. Dense Search (FAISS)
        q_vec = self.embed_query(query)
        dense_scores_raw, dense_ids_raw = self._index.search(q_vec, fetch_k)
        
        dense_map = {
            int(iid): float(score) 
            for score, iid in zip(dense_scores_raw[0], dense_ids_raw[0]) 
            if iid != -1
        }

        # 2. Sparse Search (BM25)
        sparse_map = {}
        all_sparse_scores = {}
        if self._bm25:
            tokenized_query = _tokenize(query)
            scores = self._bm25.get_scores(tokenized_query)
            for idx, score in enumerate(scores):
                all_sparse_scores[self._bm25_id_map[idx]] = float(score)
            
            # Get top fetch_k BM25 results
            top_sparse_iids = sorted(all_sparse_scores, key=all_sparse_scores.get, reverse=True)[:fetch_k]
            sparse_map = {iid: all_sparse_scores[iid] for iid in top_sparse_iids}

        # 3. Union & Filter Candidates
        candidate_ids = set(dense_map.keys()).union(set(sparse_map.keys()))
        min_dense = min(dense_map.values()) if dense_map else 0.0

        candidates = []
        for iid in candidate_ids:
            chunk = self._meta.get(iid)
            if not chunk: continue
            if section_filter and chunk.section != section_filter: continue
            if doc_index_filter is not None and chunk.doc_index != doc_index_filter: continue

            # Backfill missing scores
            d_score = dense_map.get(iid, min_dense)
            s_score = all_sparse_scores.get(iid, 0.0)
            candidates.append((iid, d_score, s_score, chunk))

        if not candidates:
            return []

        # 4. Min-Max Normalise Scores to [0, 1] range
        d_min = min(c[1] for c in candidates)
        d_max = max(c[1] for c in candidates)
        s_min = min(c[2] for c in candidates)
        s_max = max(c[2] for c in candidates)

        def _normalize(val: float, vmin: float, vmax: float) -> float:
            return (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5

        results = []
        for iid, d_score, s_score, chunk in candidates:
            nd_score = _normalize(d_score, d_min, d_max)
            ns_score = _normalize(s_score, s_min, s_max)

            # Combined score: 60% Dense, 40% Sparse
            final_score = (0.7 * nd_score) + (0.3 * ns_score)

            results.append({
                "score":        float(final_score),
                "dense_score":  float(d_score),
                "sparse_score": float(s_score),
                "chunk_id":     chunk.chunk_id,
                "doc_index":    chunk.doc_index,
                "doc_title":    chunk.doc_title,
                "section":      chunk.section,
                "doc_type":     chunk.doc_type,
                "doc_url":      chunk.doc_url,
                "chunk_index":  chunk.chunk_index,
                "text":         chunk.text,
                "raw_content":  chunk.raw_content,
            })

        # 5. Sort by hybrid score and trim to top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        logger.info("[search] returning %d result(s)  top hybrid score=%.4f", 
                    len(results), results[0]["score"] if results else 0.0)
        return results

    def get_all_chunks_for_doc(self, doc_index: int) -> list[DocumentChunk]:
        return sorted(
            [c for c in self._meta.values() if c.doc_index == doc_index],
            key=lambda c: c.chunk_index,
        )

    def stats(self) -> dict:
        sections: dict[str, int] = {}
        titles:   set[str]       = set()
        for c in self._meta.values():
            sections[c.section] = sections.get(c.section, 0) + 1
            titles.add(c.doc_title)
        return {
            "total_chunks":    len(self._meta),
            "total_documents": len(titles),
            "sections":        sections,
            "embedding_dim":   self._dim,
            "model":           self.model_name,
            "bm25_indexed":    self._bm25 is not None,
        }

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / self._INDEX_FILE))
        with open(path / self._META_FILE, "wb") as f:
            pickle.dump({"meta": self._meta, "next_id": self._next_id}, f)
        logger.info("Index saved → %s  (%d chunks)", directory, len(self._meta))

    def load(self, directory: str) -> None:
        path       = Path(directory)
        index_file = path / self._INDEX_FILE
        meta_file  = path / self._META_FILE
        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"No saved index found in {directory!r}")
        self._index = faiss.read_index(str(index_file))
        with open(meta_file, "rb") as f:
            state = pickle.load(f)
        self._meta    = state["meta"]
        self._next_id = state["next_id"]
        
        # Hydrate the BM25 index after loading metadata
        self._rebuild_bm25()
        
        logger.info("Index loaded ← %s  (%d chunks)", directory, len(self._meta))

    # -------------------------------------------------------------------------
    # Private / cleanup helpers
    # -------------------------------------------------------------------------

    def _remove_int_ids(self, int_ids: list[int]) -> int:
        if not int_ids:
            return 0
        arr = np.array(int_ids, dtype=np.int64)
        try:
            self._index.remove_ids(arr)
        except Exception as exc:
            logger.warning("FAISS remove_ids raised: %s — metadata still removed", exc)
        for iid in int_ids:
            self._meta.pop(iid, None)
            
        # Ensure BM25 stays synced with deletions
        self._rebuild_bm25()
        
        logger.info("Removed %d chunk(s) from index.", len(int_ids))
        return len(int_ids)

    def remove_docs_with_few_chunks(self, min_chunks: int = 20) -> int:
        if not self._meta or min_chunks <= 0:
            return 0
        doc_buckets: dict[int, List[int]] = {}
        for int_id, chunk in self._meta.items():
            doc_buckets.setdefault(chunk.doc_index, []).append(int_id)
        to_remove = sorted({
            iid
            for ids in doc_buckets.values() if len(ids) < min_chunks
            for iid in ids
        })
        if not to_remove:
            logger.info("remove_docs_with_few_chunks: no documents with < %d chunks.", min_chunks)
            return 0
        removed = self._remove_int_ids(to_remove)
        logger.info("remove_docs_with_few_chunks: removed %d chunks.", removed)
        return removed

    def remove_short_chunks(
        self,
        min_tokens:     int            = 20,
        min_chars:      int            = 30,
        extra_patterns: Optional[List] = None,
    ) -> int:
        if not self._meta:
            return 0
        patterns: list = []
        if extra_patterns:
            for p in extra_patterns:
                try:
                    patterns.append(re.compile(p))
                except Exception:
                    pass

        to_remove: List[int] = []
        for int_id, chunk in list(self._meta.items()):
            parts: list[str] = []
            if getattr(chunk, "text", None):
                parts.append(chunk.text)
            if getattr(chunk, "raw_content", None):
                parts.append(chunk.raw_content)
            content = "\n".join(parts).strip()

            if not content:
                to_remove.append(int_id); continue
            if min_chars and len(content) < min_chars:
                to_remove.append(int_id); continue
            if min_tokens and len(content.split()) < min_tokens:
                to_remove.append(int_id); continue
            if any(p.search(content) for p in patterns):
                to_remove.append(int_id)

        to_remove = sorted(set(to_remove))
        if not to_remove:
            logger.info("remove_short_chunks: nothing to remove.")
            return 0
        removed = self._remove_int_ids(to_remove)
        logger.info(
            "remove_short_chunks: removed %d chunk(s)  (min_tokens=%d, min_chars=%d).",
            removed, min_tokens, min_chars,
        )
        return removed