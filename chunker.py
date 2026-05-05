"""
chunker.py
----------
DocumentChunker — converts ScrapedDocuments into embedding-ready chunks.

Pipeline
--------
1. Split each document's raw content with RecursiveCharacterTextSplitter.
2. Drop chunks shorter than min_tokens (scraping artefacts: "Loading…",
   "Sign in", cookie banners, empty nav items).
3. Prepend a minimal Markdown header to each kept chunk:

       For index pages (section directory):
           ## Chunk {n} | {section}

       For content pages and videos (posts, articles, video summaries):
           ## Chunk {n} | {section} | {title}

   Title is only included when:
     - doc_type is "text" or "video_summary"  (not "index")
     - the title is a real title, not a bare URL fallback

   This is ALL that is prepended.  No URL, no doc_type, no extra noise —
   just enough for semantic grounding during retrieval.

4. Optionally deduplicate chunks across documents using cosine similarity.

Each DocumentChunk still carries full metadata (title, url, doc_type, …)
in its .metadata dict for display / citation in the UI.
"""

from __future__ import annotations

import hashlib
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from scraper import ScrapedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """One chunk ready to be embedded and stored in FAISS."""
    chunk_id: str                   # UUID string
    doc_index: int                  # Index of the source document
    doc_title: str                  # Title of the source document
    section: str                    # Section (education, blog, video, …)
    doc_url: str                    # Source URL
    doc_type: str                   # text | video_summary | link_list
    chunk_index: int                # Position of this chunk within the document
    text: str                       # Metadata header + chunk content (what gets embedded)
    raw_content: str                # Pure chunk content without the metadata header
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cosine similarity (pure Python, no dependencies)
# ---------------------------------------------------------------------------

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Normalised cosine similarity ∈ [-1, 1].
    Returns 0.0 for zero vectors.
    """
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DocumentChunker:
    """
    Converts a list of ScrapedDocuments into embedding-ready DocumentChunks.

    Parameters
    ----------
    chunk_size : int
        Maximum characters per chunk (passed to RecursiveCharacterTextSplitter).
    chunk_overlap : int
        Overlap characters between consecutive chunks.
    dedup_threshold : float | None
        If set (0–1), chunks whose text vectors exceed this cosine similarity
        to an already-seen chunk are dropped.  Requires an embedding callable
        to be passed to `chunk_documents`.
        Set to None to skip deduplication entirely.
    min_tokens : int
        Minimum whitespace-delimited token count a chunk must have to be kept.
        Chunks shorter than this are discarded — they're usually scraping
        artefacts like "Loading…", "Sign in", cookie banners, empty nav items.
        Default 30 (≈ 1–2 meaningful sentences).
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        dedup_threshold: Optional[float] = None,
        min_tokens: int = 30,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dedup_threshold = dedup_threshold
        self.min_tokens = min_tokens

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_documents(
        self,
        documents: list[ScrapedDocument],
        embed_fn=None,          # callable(text: str) -> list[float], required for dedup
    ) -> list[DocumentChunk]:
        """
        Main entry point.

        Parameters
        ----------
        documents   : list of ScrapedDocument (output of PortfolioScraper)
        embed_fn    : optional embedding function used only if dedup_threshold is set

        Returns
        -------
        list[DocumentChunk] — flat list of all chunks, sorted by (doc_index, chunk_index)
        """
        all_chunks: list[DocumentChunk] = []

        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
            logger.debug(
                "Doc [%d] '%s' → %d chunks", doc.index, doc.title, len(chunks)
            )

        if self.dedup_threshold is not None:
            if embed_fn is None:
                logger.warning(
                    "dedup_threshold is set but no embed_fn provided — skipping dedup."
                )
            else:
                before = len(all_chunks)
                all_chunks = self._deduplicate(all_chunks, embed_fn)
                logger.info(
                    "Dedup removed %d/%d chunks (threshold=%.2f)",
                    before - len(all_chunks), before, self.dedup_threshold,
                )

        return all_chunks

    def chunk_text(self, text: str) -> list[str]:
        """
        Utility: chunk arbitrary raw text without metadata injection.
        Useful for ad-hoc splitting.
        """
        return self._splitter.split_text(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_document(self, doc: ScrapedDocument) -> list[DocumentChunk]:
        """
        Split one document into kept chunks, each with a minimal Markdown header.

        Header format
        -------------
        Index page:
            ## Chunk {n} | {section}

        Content page / video summary (title is meaningful):
            ## Chunk {n} | {section} | {title}

        Content page where title is just a bare URL (fallback):
            ## Chunk {n} | {section}

        The header is the ONLY metadata prepended to the raw text before
        embedding — no source URL, no doc_type, no extra noise.
        Full metadata lives in .metadata for the retrieval UI.
        """
        raw_chunks = self._splitter.split_text(doc.content)

        # Decide whether to include the title in the header.
        # Include it when:
        #   - doc is a content page or video (not an index/directory page)
        #   - title is a real title, not the URL itself (bare-URL fallback)
        is_content = doc.doc_type in ("text", "video_summary")
        has_real_title = (
            is_content
            and doc.title
            and not doc.title.startswith(("http://", "https://"))
        )

        result: list[DocumentChunk] = []
        kept_index = 0
        dropped    = 0

        for raw in raw_chunks:
            # ── Minimum-token guard ─────────────────────────────────────
            token_count = len(raw.split())
            if token_count < self.min_tokens:
                logger.debug(
                    "Dropped short chunk (%d tokens) from doc [%d]: %r",
                    token_count, doc.index, raw[:60],
                )
                dropped += 1
                continue

            # ── Build header ────────────────────────────────────────────
            if has_real_title:
                header = f"## Chunk {kept_index + 1} | {doc.section} | {doc.title}"
            else:
                header = f"## Chunk {kept_index + 1} | {doc.section}"

            full_text = f"{header}\n\n{raw}"

            result.append(DocumentChunk(
                chunk_id    = hashlib.md5(f"{doc.url}::{doc.title}::{raw}".encode()).hexdigest(),
                doc_index   = doc.index,
                doc_title   = doc.title,
                section     = doc.section,
                doc_url     = doc.url,
                doc_type    = doc.doc_type,
                chunk_index = kept_index,
                text        = full_text,
                raw_content = raw,
                metadata    = {
                    "doc_index":    doc.index,
                    "doc_title":    doc.title,
                    "section":      doc.section,
                    "doc_url":      doc.url,
                    "doc_type":     doc.doc_type,
                    "chunk_index":  kept_index,
                    "total_chunks": len(raw_chunks),
                },
            ))
            kept_index += 1

        if dropped:
            logger.debug(
                "Doc [%d] '%s': kept %d / %d chunks (%d short-dropped)",
                doc.index, doc.title, len(result), len(raw_chunks), dropped,
            )

        return result

    def _deduplicate(
        self,
        chunks: list[DocumentChunk],
        embed_fn,
    ) -> list[DocumentChunk]:
        """
        Remove near-duplicate chunks using normalised cosine similarity.
        Keeps the first occurrence; removes later ones that are too similar.
        O(n²) — suitable for typical RAG document counts (~hundreds of chunks).
        """
        kept: list[DocumentChunk] = []
        kept_vecs: list[list[float]] = []

        for chunk in chunks:
            vec = embed_fn(chunk.text)
            is_dup = any(
                cosine_similarity(vec, kv) >= self.dedup_threshold
                for kv in kept_vecs
            )
            if not is_dup:
                kept.append(chunk)
                kept_vecs.append(vec)

        return kept